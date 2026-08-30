"""Validated, dependence-aware analysis for the reviewer revision.

The functions in this module deliberately operate only on materialized rows.
They do not import model, probe, intervention, or cache code.  Every summary
checks row status and key coverage before calculating an estimand so a failed
or missing unit can never disappear through a pandas ``dropna`` or ``groupby``.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import cache
from itertools import pairwise
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd


class AnalysisValidationError(ValueError):
    """Raised when row artifacts cannot support the prespecified analysis."""


SCORE_NULL_STATUSES = frozenset(
    ("pre_target_below_floor", "pre_control_below_floor")
)
HARD_FAILURE_STATUSES = frozenset(("failed", "invalid"))


@dataclass(frozen=True)
class CompletenessReport:
    """Exact-key coverage, including explicitly recorded failure units."""

    expected_count: int
    observed_count: int
    ok_count: int
    failure_count: int
    missing_keys: tuple[tuple[Any, ...], ...]
    unexpected_keys: tuple[tuple[Any, ...], ...]
    duplicate_keys: tuple[tuple[Any, ...], ...]
    invalid_failure_keys: tuple[tuple[Any, ...], ...]
    invalid_status_keys: tuple[tuple[Any, ...], ...]
    is_complete: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report."""

        return asdict(self)


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _stable_key_sort(keys: Iterable[tuple[Any, ...]]) -> tuple[tuple[Any, ...], ...]:
    return tuple(sorted(keys, key=lambda key: tuple(repr(item) for item in key)))


def _expected_key_tuple(item: Any, key_columns: Sequence[str]) -> tuple[Any, ...]:
    if isinstance(item, Mapping):
        missing = [column for column in key_columns if column not in item]
        if missing:
            raise AnalysisValidationError(
                f"expected-key mapping is missing columns: {missing}"
            )
        values = tuple(item[column] for column in key_columns)
    elif len(key_columns) == 1 and not isinstance(item, (tuple, list)):
        values = (item,)
    else:
        try:
            values = tuple(item)
        except TypeError as exc:
            raise AnalysisValidationError(
                f"expected key {item!r} is not iterable"
            ) from exc
        if len(values) != len(key_columns):
            raise AnalysisValidationError(
                f"expected key {item!r} has {len(values)} fields; "
                f"expected {len(key_columns)}"
            )
    if any(pd.isna(value) for value in values):
        raise AnalysisValidationError(f"expected key contains a null value: {values!r}")
    return tuple(_python_scalar(value) for value in values)


def _row_key(row: pd.Series, key_columns: Sequence[str]) -> tuple[Any, ...]:
    values = tuple(row[column] for column in key_columns)
    if any(pd.isna(value) for value in values):
        raise AnalysisValidationError(f"observed key contains a null value: {values!r}")
    return tuple(_python_scalar(value) for value in values)


def _nonblank(value: Any) -> bool:
    return not pd.isna(value) and bool(str(value).strip())


def _resolve_edit_hash_column(
    frame: pd.DataFrame,
    candidates: Sequence[str] = ("edit_hash", "epsilon_edit_hash"),
) -> str:
    ok_rows = (
        frame.loc[frame["status"].astype(str) == "ok"]
        if "status" in frame
        else frame
    )
    for candidate in candidates:
        if candidate in frame and all(_nonblank(value) for value in ok_rows[candidate]):
            return candidate
    raise AnalysisValidationError(
        "row artifact requires a nonblank paired edit_hash column"
    )


def validate_expected_keys(
    rows: pd.DataFrame | Iterable[Mapping[str, Any]],
    expected_keys: Iterable[Any],
    key_columns: Sequence[str],
    *,
    status_column: str = "status",
    success_statuses: Sequence[str] = ("ok",),
    failure_statuses: Sequence[str] = (
        "failed",
        "invalid",
        "pre_target_below_floor",
        "pre_control_below_floor",
    ),
    failure_reason_column: str = "failure_reason",
    failure_stage_column: str = "failure_stage",
    raise_on_error: bool = True,
) -> CompletenessReport:
    """Validate exact row-key coverage without treating failures as omissions.

    A failure row covers its expected key only when both ``failure_stage`` and
    ``failure_reason`` are non-empty.  It remains a failure in the returned
    report; inferential summaries reject such rows rather than excluding them.
    """

    frame = load_rows(rows)
    key_columns = tuple(key_columns)
    if not key_columns:
        raise AnalysisValidationError("key_columns must not be empty")
    missing_columns = [column for column in key_columns if column not in frame]
    if missing_columns:
        raise AnalysisValidationError(
            f"row artifact is missing key columns: {missing_columns}"
        )
    if status_column not in frame:
        raise AnalysisValidationError(
            f"row artifact is missing required status column {status_column!r}"
        )

    expected_sequence = [
        _expected_key_tuple(item, key_columns) for item in expected_keys
    ]
    expected_set = set(expected_sequence)
    if len(expected_set) != len(expected_sequence):
        duplicates = _stable_key_sort(
            key
            for key in expected_set
            if expected_sequence.count(key) > 1
        )
        raise AnalysisValidationError(f"duplicate expected keys: {duplicates!r}")

    observed_sequence = [_row_key(row, key_columns) for _, row in frame.iterrows()]
    observed_set = set(observed_sequence)
    duplicate_keys = _stable_key_sort(
        key for key in observed_set if observed_sequence.count(key) > 1
    )
    missing_keys = _stable_key_sort(expected_set - observed_set)
    unexpected_keys = _stable_key_sort(observed_set - expected_set)

    success_set = {str(status) for status in success_statuses}
    failure_set = {str(status) for status in failure_statuses}
    if not success_set or success_set & failure_set:
        raise AnalysisValidationError(
            "success and failure statuses must be non-empty and disjoint"
        )

    ok_count = 0
    failure_count = 0
    invalid_failures: set[tuple[Any, ...]] = set()
    invalid_statuses: set[tuple[Any, ...]] = set()
    for _, row in frame.iterrows():
        key = _row_key(row, key_columns)
        status = str(row[status_column])
        if status in success_set:
            ok_count += 1
        elif status in failure_set:
            failure_count += 1
            reason = row.get(failure_reason_column)
            stage = row.get(failure_stage_column)
            if not _nonblank(reason) or not _nonblank(stage):
                invalid_failures.add(key)
        else:
            invalid_statuses.add(key)

    invalid_failure_keys = _stable_key_sort(invalid_failures)
    invalid_status_keys = _stable_key_sort(invalid_statuses)
    is_complete = not any(
        (
            missing_keys,
            unexpected_keys,
            duplicate_keys,
            invalid_failure_keys,
            invalid_status_keys,
        )
    )
    report = CompletenessReport(
        expected_count=len(expected_sequence),
        observed_count=len(observed_sequence),
        ok_count=ok_count,
        failure_count=failure_count,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        duplicate_keys=duplicate_keys,
        invalid_failure_keys=invalid_failure_keys,
        invalid_status_keys=invalid_status_keys,
        is_complete=is_complete,
    )
    if raise_on_error and not is_complete:
        problems: list[str] = []
        if missing_keys:
            problems.append(f"missing expected keys: {missing_keys!r}")
        if unexpected_keys:
            problems.append(f"unexpected observed keys: {unexpected_keys!r}")
        if duplicate_keys:
            problems.append(f"duplicate observed keys: {duplicate_keys!r}")
        if invalid_failure_keys:
            problems.append(
                "failure rows require explicit failure_reason and failure_stage: "
                f"{invalid_failure_keys!r}"
            )
        if invalid_status_keys:
            problems.append(f"invalid status rows: {invalid_status_keys!r}")
        raise AnalysisValidationError("; ".join(problems))
    return report


def load_rows(
    source: pd.DataFrame | Path | str | Iterable[Mapping[str, Any]],
) -> pd.DataFrame:
    """Load an analysis row artifact without changing row order."""

    if isinstance(source, pd.DataFrame):
        return source.copy().reset_index(drop=True)
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        suffix = path.suffix.lower()
        if not path.is_file():
            raise AnalysisValidationError(f"row artifact does not exist: {path}")
        if suffix == ".csv":
            return pd.read_csv(path).reset_index(drop=True)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path).reset_index(drop=True)
        if suffix == ".jsonl":
            return pd.read_json(path, lines=True).reset_index(drop=True)
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping) and "rows" in payload:
                payload = payload["rows"]
            if not isinstance(payload, list):
                raise AnalysisValidationError(
                    f"JSON row artifact must contain a list or a 'rows' list: {path}"
                )
            return pd.DataFrame(payload).reset_index(drop=True)
        raise AnalysisValidationError(f"unsupported row artifact extension: {suffix}")
    try:
        return pd.DataFrame(list(source)).reset_index(drop=True)
    except (TypeError, ValueError) as exc:
        raise AnalysisValidationError("could not materialize rows") from exc


def _atomic_frame_write(frame: pd.DataFrame, path: Path, kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        if kind == "csv":
            frame.to_csv(temporary, index=False, lineterminator="\n")
        elif kind == "parquet":
            frame.to_parquet(temporary, index=False)
        else:  # pragma: no cover - private caller constrains this value
            raise AssertionError(kind)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_rows(
    rows: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    csv_path: Path | str,
    parquet_path: Path | str,
    key_columns: Sequence[str],
) -> pd.DataFrame:
    """Stable-sort rows by their exact key and atomically write CSV/Parquet."""

    frame = load_rows(rows)
    key_columns = tuple(key_columns)
    missing = [column for column in key_columns if column not in frame]
    if missing:
        raise AnalysisValidationError(f"row artifact is missing key columns: {missing}")
    duplicates = frame.duplicated(list(key_columns), keep=False)
    if duplicates.any():
        keys = {
            _row_key(row, key_columns)
            for _, row in frame.loc[duplicates].iterrows()
        }
        raise AnalysisValidationError(
            f"duplicate observed keys: {_stable_key_sort(keys)!r}"
        )
    condition_column = next(
        (name for name in ("condition", "scoring_condition") if name in frame),
        None,
    )
    hash_candidates_present = any(
        name in frame for name in ("edit_hash", "epsilon_edit_hash")
    )
    if condition_column is not None and hash_candidates_present:
        paired_frame = frame.loc[
            frame[condition_column].astype(str).isin(("matched", "split"))
        ].copy()
        if not paired_frame.empty:
            hash_column = _resolve_edit_hash_column(paired_frame)
            identity_columns = [
                column for column in key_columns if column != condition_column
            ]
            validate_paired_edit_hashes(
                paired_frame,
                identity_columns=identity_columns,
                condition_column=condition_column,
                edit_hash_column=hash_column,
            )
    frame = frame.sort_values(list(key_columns), kind="mergesort").reset_index(drop=True)
    _atomic_frame_write(frame, Path(csv_path), "csv")
    _atomic_frame_write(frame, Path(parquet_path), "parquet")
    return frame


def validate_paired_edit_hashes(
    rows: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    identity_columns: Sequence[str],
    condition_column: str = "condition",
    edit_hash_column: str = "edit_hash",
) -> int:
    """Fail closed unless matched/split rows identify one identical edit.

    The return value is the number of validated paired edits.  Prespecified
    score-null rows still identify the shared edit and are validated here;
    execution-failure rows without an edit are retained for the separate hard
    failure gate.
    """

    frame = load_rows(rows)
    identity_columns = tuple(identity_columns)
    _require_columns(
        frame,
        (*identity_columns, condition_column, edit_hash_column),
    )
    if not identity_columns:
        raise AnalysisValidationError("paired edit identity columns must not be empty")
    validated = 0
    grouper: str | list[str]
    grouper = list(identity_columns) if len(identity_columns) > 1 else identity_columns[0]
    for identity, group in frame.groupby(grouper, sort=True, dropna=False):
        if "status" in group and set(group["status"].astype(str)).issubset(
            HARD_FAILURE_STATUSES
        ):
            # Edit-generation failures can occur before both scoring-condition
            # rows exist and before an edit hash can be computed.  They still
            # materialize for audit and are rejected by the stage-level gate.
            continue
        conditions = {str(value) for value in group[condition_column]}
        if conditions != {"matched", "split"}:
            raise AnalysisValidationError(
                "paired edit_hash validation requires exactly matched and split "
                f"rows for {identity!r}; observed {sorted(conditions)!r}"
            )
        hashes = [value for value in group[edit_hash_column] if _nonblank(value)]
        if len(hashes) != len(group) or len({str(value) for value in hashes}) != 1:
            raise AnalysisValidationError(
                f"edit_hash mismatch between matched and split rows for {identity!r}"
            )
        validated += 1
    return validated


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise AnalysisValidationError(f"row artifact is missing columns: {missing}")


def _require_finite(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        bad = ~np.isfinite(values)
        if bad.any():
            indices = frame.index[bad].tolist()
            raise AnalysisValidationError(
                f"non-finite values in {column!r} at rows {indices[:10]}"
            )


def _resolve_column(frame: pd.DataFrame, *candidates: str) -> str:
    for candidate in candidates:
        if candidate in frame:
            return candidate
    raise AnalysisValidationError(
        f"row artifact needs one of these columns: {list(candidates)}"
    )


def _validate_status_without_expected_keys(
    frame: pd.DataFrame,
    inferred_key_columns: Sequence[str],
) -> CompletenessReport:
    keys = [
        tuple(_python_scalar(value) for value in row)
        for row in frame.loc[:, list(inferred_key_columns)].itertuples(
            index=False, name=None
        )
    ]
    return validate_expected_keys(frame, keys, inferred_key_columns)


def _validate_for_summary(
    frame: pd.DataFrame,
    *,
    expected_keys: Iterable[Any] | None,
    key_columns: Sequence[str],
    allow_score_nulls: bool = False,
) -> CompletenessReport:
    if expected_keys is None:
        report = _validate_status_without_expected_keys(frame, key_columns)
    else:
        report = validate_expected_keys(frame, expected_keys, key_columns)
    statuses = frame["status"].astype(str)
    hard_failure_count = int(statuses.isin(HARD_FAILURE_STATUSES).sum())
    score_null_count = int(statuses.isin(SCORE_NULL_STATUSES).sum())
    if hard_failure_count or (score_null_count and not allow_score_nulls):
        raise AnalysisValidationError(
            "cannot calculate inferential summary with explicit failed units: "
            f"{hard_failure_count} hard failures and {score_null_count} score-null units"
        )
    return report


def hierarchical_bootstrap(
    rows: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    value_column: str = "gap",
    block_columns: Sequence[str] = ("model", "task"),
    layer_column: str = "layer",
    pair_column: str = "pair_seed",
    draws: int = 10_000,
    seed: int = 20260830,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Three-stage percentile bootstrap for the equal-cell mean.

    Blocks are sampled first, then layers inside each sampled block, then pair
    seeds inside each sampled cell.  A cell always contributes one mean, so
    different row counts cannot accidentally change the equal-cell estimand.
    """

    if draws <= 0:
        raise AnalysisValidationError("bootstrap draws must be positive")
    if not 0.0 < confidence < 1.0:
        raise AnalysisValidationError("bootstrap confidence must be between 0 and 1")
    frame = load_rows(rows)
    block_columns = tuple(block_columns)
    required = (*block_columns, layer_column, pair_column, value_column)
    _require_columns(frame, required)
    _require_finite(frame, (value_column,))
    if frame.empty:
        raise AnalysisValidationError("cannot bootstrap an empty row artifact")
    duplicate_columns = [*block_columns, layer_column, pair_column]
    duplicate_mask = frame.duplicated(duplicate_columns, keep=False)
    if duplicate_mask.any():
        raise AnalysisValidationError(
            "hierarchical bootstrap requires one value per block/layer/pair key"
        )

    ordered = frame.sort_values(duplicate_columns, kind="mergesort")
    hierarchy: list[tuple[tuple[Any, ...], list[np.ndarray]]] = []
    grouper: str | list[str]
    grouper = list(block_columns) if len(block_columns) > 1 else block_columns[0]
    for raw_block, block_frame in ordered.groupby(grouper, sort=True):
        block = raw_block if isinstance(raw_block, tuple) else (raw_block,)
        layers: list[np.ndarray] = []
        for _, cell_frame in block_frame.groupby(layer_column, sort=True):
            values = cell_frame[value_column].to_numpy(dtype=float)
            if values.size == 0:  # pragma: no cover - groupby cannot yield empty
                raise AnalysisValidationError(f"empty cell in block {block!r}")
            layers.append(values)
        hierarchy.append((tuple(block), layers))
    if not hierarchy:
        raise AnalysisValidationError("no model-task blocks were found")

    original_cell_means = [
        float(values.mean())
        for _, layers in hierarchy
        for values in layers
    ]
    point_estimate = float(np.mean(original_cell_means))
    rng = np.random.default_rng(seed)
    replicates = np.empty(draws, dtype=np.float64)
    n_blocks = len(hierarchy)
    for draw in range(draws):
        sampled_cell_means: list[float] = []
        for block_index in rng.integers(0, n_blocks, size=n_blocks):
            layers = hierarchy[int(block_index)][1]
            n_layers = len(layers)
            for layer_index in rng.integers(0, n_layers, size=n_layers):
                values = layers[int(layer_index)]
                sampled = values[rng.integers(0, len(values), size=len(values))]
                sampled_cell_means.append(float(sampled.mean()))
        replicates[draw] = float(np.mean(sampled_cell_means))

    tail = (1.0 - confidence) / 2.0
    ci_low, ci_high = np.quantile(replicates, (tail, 1.0 - tail))
    return {
        "method": "hierarchical_percentile_bootstrap",
        "cluster_unit": "_".join(block_columns),
        "hierarchy": ["model_task", "layer", "pair"],
        "draws": int(draws),
        "seed": int(seed),
        "confidence": float(confidence),
        "point_estimate": point_estimate,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_blocks": n_blocks,
        "n_cells": len(original_cell_means),
        "n_pairs": len(frame),
    }


def exact_paired_sign_flip(
    block_deltas: Iterable[float],
    *,
    expected_blocks: int | None = 12,
    zero_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Exact two-sided sign-flip test of paired block deltas.

    Exact zeros are excluded from the sign enumeration and reported.  They
    remain in the observed mean denominator, matching the prespecified block
    estimand.  The intended primary use has at most 12 nonzero blocks.
    """

    values = np.asarray(list(block_deltas), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise AnalysisValidationError("sign-flip test requires a non-empty vector")
    if not np.isfinite(values).all():
        raise AnalysisValidationError("sign-flip deltas contain non-finite values")
    if expected_blocks is not None and len(values) != expected_blocks:
        raise AnalysisValidationError(
            f"sign-flip test expected {expected_blocks} blocks, observed {len(values)}"
        )
    if zero_tolerance < 0:
        raise AnalysisValidationError("zero_tolerance must be non-negative")
    effective = np.where(np.abs(values) > zero_tolerance, values, 0.0)
    nonzero = effective[effective != 0.0]
    zero_count = int(len(values) - len(nonzero))
    m = len(nonzero)
    exact_values = [Fraction.from_float(float(value)) for value in nonzero]
    observed_exact_sum = abs(sum(exact_values, start=Fraction(0)))
    if m == 0:
        permutations = 1
        extreme_count = 1
    else:
        permutations = 1 << m
        weights = [abs(value) for value in exact_values]
        total_weight = sum(weights, start=Fraction(0))
        if observed_exact_sum == 0:
            extreme_count = permutations
        else:
            # A signed sum is total_weight - 2 * subset_weight.  The two-sided
            # tail therefore consists of subset weights <= this threshold and
            # their complements.  Count the lower tail exactly using rational
            # representations of the input floats and branch-and-bound over
            # repeated weights.  This remains practical for the required
            # 60-cell sensitivity analysis when the observed gaps share the
            # expected direction, unlike materializing all 2**60 assignments.
            threshold = (total_weight - observed_exact_sum) / 2
            grouped = sorted(Counter(weights).items(), reverse=True)
            suffix_weight = [Fraction(0)] * (len(grouped) + 1)
            suffix_count = [0] * (len(grouped) + 1)
            for index in range(len(grouped) - 1, -1, -1):
                weight, count = grouped[index]
                suffix_weight[index] = suffix_weight[index + 1] + weight * count
                suffix_count[index] = suffix_count[index + 1] + count

            @cache
            def count_lower_tail(index: int, partial: Fraction) -> int:
                if partial > threshold:
                    return 0
                if partial + suffix_weight[index] <= threshold:
                    return 1 << suffix_count[index]
                if index == len(grouped):
                    return 1
                weight, count = grouped[index]
                total = 0
                for chosen in range(count + 1):
                    next_partial = partial + weight * chosen
                    if next_partial > threshold:
                        break
                    total += math.comb(count, chosen) * count_lower_tail(
                        index + 1, next_partial
                    )
                return total

            one_tail = count_lower_tail(0, Fraction(0))
            extreme_count = 2 * one_tail
    return {
        "method": "two_sided_exact_paired_sign_flip",
        "n_blocks": len(values),
        "nonzero_count": int(m),
        "zero_count": zero_count,
        "observed_mean": float(values.mean()),
        "permutations": int(permutations),
        "extreme_count": int(extreme_count),
        "p_value": float(extreme_count / permutations),
        "zero_tolerance": float(zero_tolerance),
    }


def _sign_counts(values: pd.Series | np.ndarray) -> dict[str, int]:
    array = np.asarray(values, dtype=float)
    return {
        "positive": int(np.count_nonzero(array > 0.0)),
        "zero": int(np.count_nonzero(array == 0.0)),
        "negative": int(np.count_nonzero(array < 0.0)),
    }


def _distribution_summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise AnalysisValidationError("distribution summary requires finite values")
    if not np.isfinite(array).all():
        raise AnalysisValidationError("distribution summary contains non-finite values")
    ordered = np.sort(array)
    q25, median, q75 = np.quantile(ordered, (0.25, 0.5, 0.75))
    return {
        "n": len(ordered),
        "mean": float(ordered.mean()),
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "values": [float(value) for value in ordered],
    }


def _common_summary_metadata(
    *,
    raw_row_file_sha256: str | None,
    generating_git_commit: str | None,
    warnings: Sequence[str] | None,
    caveats: Sequence[str] | None,
) -> dict[str, Any]:
    return {
        "raw_row_file_sha256": raw_row_file_sha256,
        "generating_git_commit": generating_git_commit,
        "warnings": list(warnings or ()),
        "caveats": list(caveats or ()),
    }


def summarize_matched_split(
    rows: pd.DataFrame | Path | str | Iterable[Mapping[str, Any]],
    *,
    expected_keys: Iterable[Any] | None = None,
    key_columns: Sequence[str] = (
        "model",
        "task",
        "layer",
        "pair_seed",
        "method",
        "condition",
    ),
    primary_method: str = "alterrep",
    expected_blocks: int = 12,
    bootstrap_draws: int = 10_000,
    bootstrap_seed: int = 20260830,
    permutation_seed: int = 20260831,
    raw_row_file_sha256: str | None = None,
    generating_git_commit: str | None = None,
    warnings: Sequence[str] | None = None,
    caveats: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Summarize matched-versus-split damage with block-aware inference."""

    frame = load_rows(rows)
    _validate_for_summary(
        frame,
        expected_keys=expected_keys,
        key_columns=key_columns,
        allow_score_nulls=True,
    )
    model_col = _resolve_column(frame, "model", "model_key")
    task_col = _resolve_column(frame, "task", "task_key")
    method_col = _resolve_column(frame, "method")
    condition_col = _resolve_column(frame, "condition", "scoring_condition")
    damage_col = _resolve_column(frame, "target_damage_C", "C", "target_damage")
    edit_hash_col = _resolve_edit_hash_column(frame, ("edit_hash",))
    _require_columns(frame, ("layer", "pair_seed"))

    primary_requested = frame.loc[
        frame[method_col].astype(str) == primary_method
    ].copy()
    if primary_requested.empty:
        raise AnalysisValidationError(
            f"no rows found for primary method {primary_method!r}"
        )
    pair_columns = [model_col, task_col, "layer", "pair_seed"]
    duplicate_mask = primary_requested.duplicated(
        [*pair_columns, condition_col], keep=False
    )
    if duplicate_mask.any():
        raise AnalysisValidationError(
            "matched/split rows contain duplicate pair-condition keys"
        )
    condition_sets = primary_requested.groupby(
        pair_columns, sort=True, dropna=False
    )[condition_col].agg(lambda values: frozenset(str(value) for value in values))
    invalid_conditions = condition_sets[
        condition_sets != frozenset(("matched", "split"))
    ]
    if not invalid_conditions.empty:
        raise AnalysisValidationError(
            "every primary pair must contain exactly matched and split rows; "
            f"invalid pairs: {list(invalid_conditions.index[:10])!r}"
        )
    validated_edit_pairs = validate_paired_edit_hashes(
        primary_requested,
        identity_columns=pair_columns,
        condition_column=condition_col,
        edit_hash_column=edit_hash_col,
    )

    complete_pair_mask = primary_requested.groupby(
        pair_columns, sort=False, dropna=False
    )["status"].transform(lambda values: bool((values.astype(str) == "ok").all()))
    primary = primary_requested.loc[complete_pair_mask].copy()
    if primary.empty:
        raise AnalysisValidationError(
            "no complete matched/split pairs remain after applying the locked score floor"
        )
    _require_finite(primary, (damage_col,))
    requested_pair_count = len(condition_sets)
    analyzed_pair_count = int(
        primary[pair_columns].drop_duplicates().shape[0]
    )
    score_null_primary = primary_requested.loc[
        primary_requested["status"].astype(str).isin(SCORE_NULL_STATUSES)
    ]
    excluded_pair_keys = [
        tuple(_python_scalar(value) for value in key)
        for key in primary_requested.loc[~complete_pair_mask, pair_columns]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ]

    paired = (
        primary.set_index([*pair_columns, condition_col])[damage_col]
        .unstack(condition_col)
        .reset_index()
    )
    paired["gap"] = paired["matched"] - paired["split"]
    cell_columns = [model_col, task_col, "layer"]
    cells = (
        paired.groupby(cell_columns, sort=True, as_index=False)[
            ["matched", "split", "gap"]
        ]
        .mean()
    )
    requested_cells = primary_requested[cell_columns].drop_duplicates()
    analyzed_cell_keys = set(
        cells[cell_columns].itertuples(index=False, name=None)
    )
    excluded_cell_keys = [
        tuple(_python_scalar(value) for value in key)
        for key in requested_cells.itertuples(index=False, name=None)
        if tuple(key) not in analyzed_cell_keys
    ]
    if "depth_position" in primary_requested:
        depth_counts = primary_requested.groupby(cell_columns)["depth_position"].nunique()
        if (depth_counts != 1).any():
            raise AnalysisValidationError("depth_position changes within a cell")
        depths = (
            primary_requested.groupby(cell_columns, as_index=False)["depth_position"]
            .first()
        )
        cells = cells.merge(depths, on=cell_columns, validate="one_to_one")
    else:
        cells["depth_position"] = 0
        for _, block in cells.groupby([model_col, task_col], sort=True):
            ordered_layers = sorted(block["layer"].unique().tolist())
            positions = {layer: index + 1 for index, layer in enumerate(ordered_layers)}
            cells.loc[block.index, "depth_position"] = block["layer"].map(positions)

    blocks = (
        cells.groupby([model_col, task_col], sort=True, as_index=False)[
            ["matched", "split", "gap"]
        ]
        .mean()
    )
    if len(blocks) != expected_blocks:
        raise AnalysisValidationError(
            f"matched/split summary expected {expected_blocks} model-task blocks, "
            f"observed {len(blocks)}"
        )
    bootstrap_frame = paired.rename(
        columns={model_col: "model", task_col: "task"}
    )
    bootstrap = hierarchical_bootstrap(
        bootstrap_frame,
        draws=bootstrap_draws,
        seed=bootstrap_seed,
    )
    sign_flip = exact_paired_sign_flip(
        blocks["gap"].to_numpy(), expected_blocks=expected_blocks
    )
    secondary_sign_flip = exact_paired_sign_flip(
        cells["gap"].to_numpy(), expected_blocks=len(cells)
    )
    secondary_sign_flip.update(
        {
            "n_cells": len(cells),
            "unit": "model_layer_task_cell",
            "sensitivity_only": True,
            "dependence_caveat": "layers within a model-task block share examples",
        }
    )
    cell_signs = _sign_counts(cells["gap"])
    block_signs = _sign_counts(blocks["gap"])
    n_cells = len(cells)
    n_requested_cells = len(requested_cells)
    n_blocks = len(blocks)

    depth_means = {
        str(_python_scalar(depth)): float(value)
        for depth, value in cells.groupby("depth_position", sort=True)["gap"].mean().items()
    }
    depth_medians = {
        str(_python_scalar(depth)): float(value)
        for depth, value in cells.groupby("depth_position", sort=True)["gap"].median().items()
    }
    task_means = {
        str(task): float(value)
        for task, value in cells.groupby(task_col, sort=True)["gap"].mean().items()
    }
    leave_one_model_out_means = {
        str(model): float(cells.loc[cells[model_col] != model, "gap"].mean())
        for model in sorted(cells[model_col].unique(), key=str)
    }
    leave_one_model_out: dict[str, dict[str, Any]] = {}
    for offset, model in enumerate(sorted(cells[model_col].unique(), key=str), start=1):
        reduced_pairs = paired.loc[paired[model_col] != model].rename(
            columns={model_col: "model", task_col: "task"}
        )
        reduced_bootstrap = hierarchical_bootstrap(
            reduced_pairs,
            draws=bootstrap_draws,
            seed=bootstrap_seed + offset,
        )
        leave_one_model_out[str(model)] = {
            "mean": leave_one_model_out_means[str(model)],
            "ci_low": reduced_bootstrap["ci_low"],
            "ci_high": reduced_bootstrap["ci_high"],
            "method": reduced_bootstrap["method"],
            "draws": reduced_bootstrap["draws"],
            "seed": reduced_bootstrap["seed"],
            "model_task_blocks": reduced_bootstrap["n_blocks"],
        }

    pre_accuracy_col = next(
        (
            name
            for name in (
                "target_accuracy_pre",
                "target_acc_pre",
                "raw_target_accuracy_pre",
            )
            if name in primary_requested
        ),
        None,
    )
    post_accuracy_col = next(
        (
            name
            for name in (
                "target_accuracy_post",
                "target_acc_post",
                "raw_target_accuracy_post",
            )
            if name in primary_requested
        ),
        None,
    )
    if (pre_accuracy_col is None) != (post_accuracy_col is None):
        raise AnalysisValidationError(
            "matched/split rows must provide both pre and post target accuracy"
        )
    if pre_accuracy_col is None:
        target_accuracy_distributions: dict[str, Any] = {
            "available": False,
            "reason": "pre/post target accuracy columns are absent",
        }
    else:
        _require_finite(
            primary_requested, (pre_accuracy_col, post_accuracy_col)
        )
        target_accuracy_distributions = {"available": True, "unit": "pair_row"}
        for condition in ("matched", "split"):
            condition_rows = primary_requested.loc[
                primary_requested[condition_col].astype(str) == condition
            ]
            target_accuracy_distributions[condition] = {
                "pre": _distribution_summary(condition_rows[pre_accuracy_col]),
                "post": _distribution_summary(condition_rows[post_accuracy_col]),
            }
    grand_matched = float(cells["matched"].mean())
    grand_split = float(cells["split"].mean())
    grand_gap = float(cells["gap"].mean())
    summary: dict[str, Any] = {
        "schema_version": 1,
        "estimand": (
            "Available-case paired contrast: for every pair with defined matched "
            "and split target damage, target_damage_C_matched minus "
            "target_damage_C_split on the identical edit; average complete pairs "
            "within estimable cells and estimable selected layers within each "
            "equally weighted model-task block. Requested score-null rows and "
            "non-estimable cells remain explicit."
        ),
        "primary_estimand_status": (
            "available_case" if excluded_cell_keys else "fully_observed"
        ),
        "included_units": {
            "rows": len(primary_requested),
            "pairs": requested_pair_count,
            "cells": int(n_requested_cells),
            "model_task_blocks": int(n_blocks),
        },
        "analyzed_units": {
            "rows": len(primary),
            "pairs": analyzed_pair_count,
            "cells": int(n_cells),
            "model_task_blocks": int(n_blocks),
        },
        "validated_paired_edit_hashes": int(validated_edit_pairs),
        "failed_units": int(
            frame["status"].astype(str).isin(HARD_FAILURE_STATUSES).sum()
        ),
        "score_null_units": int(
            frame["status"].astype(str).isin(SCORE_NULL_STATUSES).sum()
        ),
        "primary_score_null_units": len(score_null_primary),
        "primary_score_null_pairs": requested_pair_count - analyzed_pair_count,
        "primary_score_null_cells": len(excluded_cell_keys),
        "primary_score_null_status_counts": dict(
            Counter(score_null_primary["status"].astype(str))
        ),
        "excluded_primary_pair_keys": excluded_pair_keys,
        "excluded_primary_cell_keys": excluded_cell_keys,
        "grand_mean_matched": grand_matched,
        "grand_mean_split": grand_split,
        "grand_mean_gap": grand_gap,
        "median_gap": float(cells["gap"].median()),
        "fraction_cell_gaps_positive": float(cell_signs["positive"] / n_cells),
        "fraction_block_gaps_positive": float(block_signs["positive"] / n_blocks),
        "cell_gap_sign_counts": cell_signs,
        "block_gap_sign_counts": block_signs,
        "depth_position_means": depth_means,
        "depth_position_medians": depth_medians,
        "task_specific_means": task_means,
        "leave_one_model_out_means": leave_one_model_out_means,
        "leave_one_model_out": leave_one_model_out,
        "target_accuracy_distributions": target_accuracy_distributions,
        "primary_exact_sign_flip": sign_flip,
        "secondary_exact_sign_flip": secondary_sign_flip,
        "bootstrap": bootstrap,
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_draws": int(bootstrap_draws),
        "permutation_seed": int(permutation_seed),
        "confidence_interval_method": "hierarchical_percentile_bootstrap",
        "cluster_unit": "model_task",
        "manuscript_values": {
            "matched": f"{grand_matched:.3f}",
            "split": f"{grand_split:.3f}",
            "gap": f"{grand_gap:.3f}",
            "ci": f"[{bootstrap['ci_low']:.3f}, {bootstrap['ci_high']:.3f}]",
            "p": f"{sign_flip['p_value']:.6f}",
        },
    }
    summary.update(
        _common_summary_metadata(
            raw_row_file_sha256=raw_row_file_sha256,
            generating_git_commit=generating_git_commit,
            warnings=warnings,
            caveats=(
                caveats
                or (
                    (
                        f"{n_cells} model-layer-task cells are nested in "
                        f"{n_blocks} model-task blocks."
                    ),
                    (
                        f"{requested_pair_count - analyzed_pair_count} prespecified "
                        "pairs had a denominator below the locked floor; both "
                        "conditions for each affected pair are excluded from the "
                        "paired estimand and remain explicit in the row artifacts."
                    ),
                    (
                        f"{len(excluded_cell_keys)} of {n_requested_cells} requested "
                        "cells have no complete pair and are explicitly non-estimable; "
                        f"the reported available-case summaries use {n_cells} cells "
                        f"while retaining all {n_blocks} model-task blocks."
                    ),
                    "Fixed-decoder target damage is not evidence of erasure.",
                )
            ),
        )
    )
    return summary


def summarize_epsilon_sweep(
    rows: pd.DataFrame | Path | str | Iterable[Mapping[str, Any]],
    *,
    expected_keys: Iterable[Any] | None = None,
    key_columns: Sequence[str] = (
        "model",
        "task",
        "layer",
        "pair_seed",
        "method",
        "condition",
        "epsilon",
    ),
    zero_tolerance: float = 1.0e-6,
    norm_tolerance: float = 1.0e-6,
    nonmonotonic_reversal_threshold: float = 0.10,
    raw_row_file_sha256: str | None = None,
    generating_git_commit: str | None = None,
    warnings: Sequence[str] | None = None,
    caveats: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Summarize the complete paired FGSM/PGD epsilon grid."""

    frame = load_rows(rows)
    _validate_for_summary(
        frame,
        expected_keys=expected_keys,
        key_columns=key_columns,
        allow_score_nulls=True,
    )
    model_col = _resolve_column(frame, "model", "model_key")
    task_col = _resolve_column(frame, "task", "task_key")
    method_col = _resolve_column(frame, "method")
    condition_col = _resolve_column(frame, "condition", "scoring_condition")
    epsilon_col = _resolve_column(frame, "epsilon", "eps")
    damage_col = _resolve_column(frame, "target_damage_C", "C", "target_damage")
    edit_hash_col = _resolve_edit_hash_column(frame)
    _require_columns(frame, ("layer", "pair_seed"))
    if nonmonotonic_reversal_threshold < 0:
        raise AnalysisValidationError(
            "nonmonotonic reversal threshold must be non-negative"
        )

    requested_frame = frame.copy()
    paired_keys = [
        model_col,
        task_col,
        "layer",
        "pair_seed",
        method_col,
        epsilon_col,
    ]
    condition_duplicates = requested_frame.duplicated(
        [*paired_keys, condition_col], keep=False
    )
    if condition_duplicates.any():
        raise AnalysisValidationError("epsilon rows contain duplicate condition keys")
    condition_sets = requested_frame.groupby(
        paired_keys, sort=True, dropna=False
    )[condition_col].agg(lambda values: frozenset(str(value) for value in values))
    if (condition_sets != frozenset(("matched", "split"))).any():
        raise AnalysisValidationError(
            "every epsilon unit must contain exactly matched and split rows"
        )
    validated_edit_pairs = validate_paired_edit_hashes(
        requested_frame,
        identity_columns=paired_keys,
        condition_column=condition_col,
        edit_hash_column=edit_hash_col,
    )
    complete_pair_mask = requested_frame.groupby(
        paired_keys, sort=False, dropna=False
    )["status"].transform(lambda values: bool((values.astype(str) == "ok").all()))
    frame = requested_frame.loc[complete_pair_mask].copy()
    if frame.empty:
        raise AnalysisValidationError(
            "no complete epsilon pairs remain after applying the locked score floor"
        )
    score_null_rows = requested_frame.loc[
        requested_frame["status"].astype(str).isin(SCORE_NULL_STATUSES)
    ]

    metric_aliases = {
        "target_damage_C": damage_col,
        "control_preservation_S": next(
            (name for name in ("control_preservation_S", "S") if name in frame),
            None,
        ),
        "H": next((name for name in ("H", "harmonic_mean") if name in frame), None),
        "target_accuracy_pre": next(
            (name for name in ("target_accuracy_pre", "target_acc_pre") if name in frame),
            None,
        ),
        "target_accuracy_post": next(
            (name for name in ("target_accuracy_post", "target_acc_post") if name in frame),
            None,
        ),
        "control_accuracy_pre": next(
            (name for name in ("control_accuracy_pre", "control_acc_pre") if name in frame),
            None,
        ),
        "control_accuracy_post": next(
            (name for name in ("control_accuracy_post", "control_acc_post") if name in frame),
            None,
        ),
        "orientation_sensitivity_accuracy": next(
            (
                name
                for name in ("orientation_sensitivity_accuracy", "max_accuracy_sensitivity")
                if name in frame
            ),
            None,
        ),
        "auc": "auc" if "auc" in frame else None,
        "log_loss": "log_loss" if "log_loss" in frame else None,
        "realized_linf_norm": (
            "realized_linf_norm" if "realized_linf_norm" in frame else None
        ),
    }
    present_metrics = {
        canonical: source
        for canonical, source in metric_aliases.items()
        if source is not None
    }
    _require_finite(frame, tuple(present_metrics.values()) + (epsilon_col,))
    if ((frame[damage_col] < 0.0) | (frame[damage_col] > 1.0)).any():
        raise AnalysisValidationError("target_damage_C must lie in [0, 1]")

    if "realized_linf_norm" in present_metrics:
        norm_source = present_metrics["realized_linf_norm"]
        _require_finite(requested_frame, (norm_source, epsilon_col))
        norms = requested_frame[norm_source].to_numpy(dtype=float)
        epsilons = requested_frame[epsilon_col].to_numpy(dtype=float)
        if (norms < -norm_tolerance).any() or (norms > epsilons + norm_tolerance).any():
            raise AnalysisValidationError(
                "realized perturbation violates the requested L-infinity bound"
            )

    cell_columns = [
        model_col,
        task_col,
        "layer",
        method_col,
        condition_col,
        epsilon_col,
    ]
    requested_cells = requested_frame[cell_columns].drop_duplicates()
    cells = (
        frame.groupby(cell_columns, sort=True, as_index=False)[
            list(present_metrics.values())
        ]
        .mean()
        .rename(columns={source: canonical for canonical, source in present_metrics.items()})
    )
    analyzed_curve_cell_keys = set(
        cells[cell_columns].itertuples(index=False, name=None)
    )
    excluded_curve_cell_keys = [
        tuple(_python_scalar(value) for value in key)
        for key in requested_cells.itertuples(index=False, name=None)
        if tuple(key) not in analyzed_curve_cell_keys
    ]
    requested_blocks = int(
        requested_frame[[model_col, task_col]].drop_duplicates().shape[0]
    )
    analyzed_blocks = int(
        frame[[model_col, task_col]].drop_duplicates().shape[0]
    )
    if analyzed_blocks != requested_blocks:
        raise AnalysisValidationError(
            "the locked score floor removed every estimable epsilon unit from a "
            "requested model-task block"
        )
    curves: list[dict[str, Any]] = []
    for (method, condition, epsilon), group in cells.groupby(
        [method_col, condition_col, epsilon_col], sort=True
    ):
        pair_rows = frame.loc[
            (frame[method_col].astype(str) == str(method))
            & (frame[condition_col].astype(str) == str(condition))
            & np.isclose(
                frame[epsilon_col].to_numpy(dtype=float),
                float(epsilon),
                rtol=0.0,
                atol=0.0,
            )
        ]
        row: dict[str, Any] = {
            "method": str(method),
            "condition": str(condition),
            "epsilon": float(epsilon),
            "n_cells": len(group),
            "n_pairs": len(pair_rows),
            "fraction_at_C_equal_1": float(
                np.mean(
                    np.isclose(
                        pair_rows[damage_col].to_numpy(dtype=float),
                        1.0,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                )
            ),
        }
        for metric in present_metrics:
            row[f"mean_{metric}"] = float(group[metric].mean())
        curves.append(row)

    zero_requested = requested_frame.loc[
        np.isclose(requested_frame[epsilon_col].to_numpy(dtype=float), 0.0)
    ].copy()
    zero = frame.loc[
        np.isclose(frame[epsilon_col].to_numpy(dtype=float), 0.0)
    ].copy()
    if zero.empty:
        raise AnalysisValidationError("epsilon sweep contains no epsilon-zero rows")
    max_damage = float(np.max(np.abs(zero[damage_col].to_numpy(dtype=float))))
    norm_column = present_metrics.get("realized_linf_norm")
    max_norm = (
        float(np.max(np.abs(zero_requested[norm_column].to_numpy(dtype=float))))
        if norm_column is not None
        else 0.0
    )
    pre_source = present_metrics.get("target_accuracy_pre")
    post_source = present_metrics.get("target_accuracy_post")
    max_accuracy_noop_difference = 0.0
    if pre_source is not None and post_source is not None:
        _require_finite(zero_requested, (pre_source, post_source))
        max_accuracy_noop_difference = float(
            np.max(
                np.abs(
                    zero_requested[pre_source].to_numpy(dtype=float)
                    - zero_requested[post_source].to_numpy(dtype=float)
                )
            )
        )
    zero_passed = (
        max_damage <= zero_tolerance
        and max_norm <= norm_tolerance
        and max_accuracy_noop_difference <= zero_tolerance
    )
    if not zero_passed:
        raise AnalysisValidationError(
            "epsilon-zero integrity failed: expected an exact no-op within tolerance"
        )

    paired = (
        frame.set_index([*paired_keys, condition_col])[damage_col]
        .unstack(condition_col)
        .reset_index()
    )
    paired["gap"] = paired["matched"] - paired["split"]
    gap_curves = [
        {
            "method": str(method),
            "epsilon": float(epsilon),
            "mean_gap": float(group["gap"].mean()),
            "n_pairs": len(group),
        }
        for (method, epsilon), group in paired.groupby(
            [method_col, epsilon_col], sort=True
        )
    ]
    large_reversals: list[dict[str, Any]] = []
    for (method, condition), group in pd.DataFrame(curves).groupby(
        ["method", "condition"], sort=True
    ):
        ordered_curve = group.sort_values("epsilon", kind="mergesort")
        records = ordered_curve.to_dict(orient="records")
        for previous, current in pairwise(records):
            reversal = (
                float(previous["mean_target_damage_C"])
                - float(current["mean_target_damage_C"])
            )
            if reversal > nonmonotonic_reversal_threshold:
                large_reversals.append(
                    {
                        "method": str(method),
                        "condition": str(condition),
                        "epsilon_from": float(previous["epsilon"]),
                        "epsilon_to": float(current["epsilon"]),
                        "mean_C_from": float(previous["mean_target_damage_C"]),
                        "mean_C_to": float(current["mean_target_damage_C"]),
                        "reversal_size": float(reversal),
                    }
                )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "estimand": (
            "Available-case curves: at every prespecified epsilon, average "
            "complete-pair values within each estimable model-layer-task cell "
            "and then weight estimable cells equally, separately for FGSM/PGD "
            "and matched/split scoring; retain all requested score-null rows and "
            "non-estimable curve cells explicitly."
        ),
        "estimand_status": (
            "available_case" if excluded_curve_cell_keys else "fully_observed"
        ),
        "included_units": {
            "rows": len(requested_frame),
            "cells": len(requested_cells),
            "model_task_blocks": int(
                requested_blocks
            ),
        },
        "analyzed_units": {
            "rows": len(frame),
            "paired_units": int(frame[paired_keys].drop_duplicates().shape[0]),
            "cells": len(cells),
            "model_task_blocks": int(
                analyzed_blocks
            ),
        },
        "validated_paired_edit_hashes": int(validated_edit_pairs),
        "failed_units": int(
            requested_frame["status"].astype(str).isin(HARD_FAILURE_STATUSES).sum()
        ),
        "score_null_units": len(score_null_rows),
        "score_null_pairs": len(condition_sets)
        - int(frame[paired_keys].drop_duplicates().shape[0]),
        "score_null_curve_cells": len(excluded_curve_cell_keys),
        "excluded_curve_cell_keys": excluded_curve_cell_keys,
        "score_null_status_counts": dict(
            Counter(score_null_rows["status"].astype(str))
        ),
        "curves": curves,
        "paired_gap_curves": gap_curves,
        "large_nonmonotonic_reversals": large_reversals,
        "has_large_nonmonotonic_reversals": bool(large_reversals),
        "nonmonotonic_reversal_threshold": float(
            nonmonotonic_reversal_threshold
        ),
        "epsilon_zero_integrity": {
            "row_count": len(zero_requested),
            "analyzed_damage_row_count": len(zero),
            "score_null_row_count": len(zero_requested) - len(zero),
            "max_abs_target_damage_C": max_damage,
            "max_realized_linf_norm": max_norm,
            "max_abs_target_accuracy_noop_difference": max_accuracy_noop_difference,
            "passed": bool(zero_passed),
        },
        "confidence_interval_method": "model_task_cluster_bootstrap_for_figures",
        "cluster_unit": "model_task",
    }
    summary.update(
        _common_summary_metadata(
            raw_row_file_sha256=raw_row_file_sha256,
            generating_git_commit=generating_git_commit,
            warnings=warnings,
            caveats=(
                caveats
                or (
                    "The complete prespecified epsilon grid is reported; no epsilon is selected post hoc.",
                    (
                        f"{len(condition_sets) - int(frame[paired_keys].drop_duplicates().shape[0])} "
                        "paired epsilon units had a denominator below the locked "
                        "floor; both scoring conditions remain explicit and are "
                        "excluded symmetrically from curve estimates."
                    ),
                    (
                        f"{len(excluded_curve_cell_keys)} of {len(requested_cells)} "
                        "requested method-condition-epsilon curve cells are "
                        "non-estimable; available-case curves retain all "
                        f"{analyzed_blocks} model-task blocks."
                    ),
                    "Fixed-decoder target damage is not evidence of erasure.",
                )
            ),
        )
    )
    return summary


def summarize_construct_check(
    rows: pd.DataFrame | Path | str | Iterable[Mapping[str, Any]],
    *,
    expected_keys: Iterable[Any] | None = None,
    key_columns: Sequence[str] = ("edit_id", "evaluation_family", "label"),
    expected_edit_ids: Iterable[str] | None = None,
    inversion_threshold: float = 0.5,
    redecodable_threshold: float = 0.5,
    raw_row_file_sha256: str | None = None,
    generating_git_commit: str | None = None,
    warnings: Sequence[str] | None = None,
    caveats: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Summarize fixed-orientation and fresh-decoder construct checks."""

    frame = load_rows(rows)
    report = _validate_for_summary(
        frame,
        expected_keys=expected_keys,
        key_columns=key_columns,
    )
    _require_columns(frame, ("edit_id", "edit_object", "evaluation_family", "label"))
    observed_edit_ids = {str(value) for value in frame["edit_id"].unique()}
    if expected_edit_ids is not None:
        expected_edit_set = {str(value) for value in expected_edit_ids}
        missing_edits = expected_edit_set - observed_edit_ids
        unexpected_edits = observed_edit_ids - expected_edit_set
        if missing_edits or unexpected_edits:
            raise AnalysisValidationError(
                "construct edit coverage mismatch: "
                f"missing={sorted(missing_edits)!r}, "
                f"unexpected={sorted(unexpected_edits)!r}"
            )

    numeric_columns = [
        column
        for column in (
            "accuracy",
            "balanced_accuracy",
            "calibrated_orientation_accuracy",
            "max_accuracy_sensitivity",
            "auc",
            "log_loss",
            "orientation_adjusted_auc",
            "orientation_adjusted_log_loss",
            "C_raw",
            "C_orientation_calibrated",
            "target_recovery_ratio",
            "control_retention_ratio",
        )
        if column in frame
    ]
    if "accuracy" not in numeric_columns:
        raise AnalysisValidationError("construct rows require raw accuracy")
    ratio_labels = {
        "target_recovery_ratio": "target",
        "control_retention_ratio": "control",
    }
    shared_numeric_columns = [
        column for column in numeric_columns if column not in ratio_labels
    ]
    _require_finite(frame, shared_numeric_columns)
    for column, applicable_label in ratio_labels.items():
        if column not in frame:
            continue
        applicable_rows = frame.loc[
            frame["label"].astype(str) == applicable_label
        ]
        if applicable_rows.empty:
            raise AnalysisValidationError(
                f"construct rows contain no {applicable_label!r} values for {column!r}"
            )
        _require_finite(applicable_rows, (column,))

    def numeric_columns_for_label(label: str) -> list[str]:
        return [
            column
            for column in numeric_columns
            if column not in ratio_labels or ratio_labels[column] == label
        ]

    aggregate_columns = ["edit_object", "evaluation_family", "label"]
    aggregates: list[dict[str, Any]] = []
    for keys, group in frame.groupby(aggregate_columns, sort=True):
        edit_object, evaluation_family, label = keys
        record: dict[str, Any] = {
            "edit_object": str(edit_object),
            "evaluation_family": str(evaluation_family),
            "label": str(label),
            "n_rows": len(group),
        }
        for column in numeric_columns_for_label(str(label)):
            record[f"mean_{column}"] = float(group[column].mean())
        aggregates.append(record)

    fixed_target = frame.loc[
        (frame["evaluation_family"].astype(str) == "fixed")
        & (frame["label"].astype(str) == "target")
    ]
    orientation_column = (
        "calibrated_orientation_accuracy"
        if "calibrated_orientation_accuracy" in frame
        else None
    )
    if orientation_column is None:
        inversion_ids: set[str] = set()
    else:
        inversion_ids = {
            str(value)
            for value in fixed_target.loc[
                (fixed_target["accuracy"] < inversion_threshold)
                & (fixed_target[orientation_column] > inversion_threshold),
                "edit_id",
            ]
        }
    fresh_target = frame.loc[
        frame["evaluation_family"].astype(str).str.startswith("fresh")
        & (frame["label"].astype(str) == "target")
    ]
    redecodable = fresh_target["accuracy"] > redecodable_threshold
    redecodable_ids = sorted(
        {
            str(value)
            for value in fresh_target.loc[redecodable, "edit_id"]
        }
    )

    if "candidate_architecture" in frame:
        architecture_values = frame["candidate_architecture"]
        candidate_mask = architecture_values.map(_nonblank)
    elif "architecture" in frame:
        architecture_values = frame["architecture"]
        if (frame["edit_object"].astype(str) == "dcand_crossfit").any():
            candidate_mask = frame["edit_object"].astype(str) == "dcand_crossfit"
        else:
            candidate_mask = frame["edit_object"].astype(str) != "alterrep"
    else:
        architecture_values = frame["edit_object"]
        candidate_mask = frame["edit_object"].astype(str) != "alterrep"
    candidates = frame.loc[candidate_mask].copy()
    candidates["_candidate_architecture"] = architecture_values.loc[
        candidate_mask
    ].astype(str)
    candidate_groups: list[dict[str, Any]] = []
    if not candidates.empty:
        per_candidate = (
            candidates.groupby(
                [
                    "edit_id",
                    "_candidate_architecture",
                    "evaluation_family",
                    "label",
                ],
                sort=True,
                as_index=False,
            )[numeric_columns]
            .mean()
        )
        for keys, group in per_candidate.groupby(
            ["_candidate_architecture", "evaluation_family", "label"],
            sort=True,
        ):
            architecture, evaluation_family, label = keys
            candidate_groups.append(
                {
                    "candidate_architecture": str(architecture),
                    "evaluation_family": str(evaluation_family),
                    "label": str(label),
                    "candidate_count": int(group["edit_id"].nunique()),
                    "edit_ids": sorted(str(value) for value in group["edit_id"]),
                    "metrics": {
                        column: _distribution_summary(group[column])
                        for column in numeric_columns_for_label(str(label))
                    },
                }
            )
    candidate_distribution_summary = {
        "unit": "unique_candidate_edit",
        "candidate_edit_count": int(candidates["edit_id"].nunique()),
        "groups": candidate_groups,
    }

    summary: dict[str, Any] = {
        "schema_version": 1,
        "estimand": (
            "For each prespecified edit, report fixed-evaluator raw and "
            "calibration-frozen orientation metrics plus independently trained "
            "fresh-decoder accuracy on the untouched final-test split."
        ),
        "included_units": {
            "rows": len(frame),
            "edits": len(observed_edit_ids),
        },
        "failed_units": int(report.failure_count),
        "aggregates": aggregates,
        "inversion_count": len(inversion_ids),
        "inversion_edit_ids": sorted(inversion_ids),
        "redecodable_count": len(redecodable_ids),
        "redecodable_edit_ids": redecodable_ids,
        "redecodable_family_rows": int(redecodable.sum()),
        "fresh_target_evaluations": len(fresh_target),
        "candidate_distribution_summary": candidate_distribution_summary,
        "orientation_choice_split": "orientation_calibration",
        "orientation_application_split": "final_test",
        "fresh_decoder_fit_split": "fresh_decoder_fit",
        "fresh_decoder_evaluation_split": "final_test",
    }
    summary.update(
        _common_summary_metadata(
            raw_row_file_sha256=raw_row_file_sha256,
            generating_git_commit=generating_git_commit,
            warnings=warnings,
            caveats=(
                caveats
                or (
                    "This is one prespecified model-task-layer cell.",
                    "Absence of fresh-decoder recovery in one cell is not proof of erasure.",
                )
            ),
        )
    )
    return summary


__all__ = [
    "AnalysisValidationError",
    "CompletenessReport",
    "exact_paired_sign_flip",
    "hierarchical_bootstrap",
    "load_rows",
    "materialize_rows",
    "summarize_construct_check",
    "summarize_epsilon_sweep",
    "summarize_matched_split",
    "validate_expected_keys",
    "validate_paired_edit_hashes",
]
