"""Post-hoc denominator-floor robustness analyses for the reviewer revision.

These summaries are derived only from materialized matched/split rows.  They
are sensitivity analyses and never replace the registered normalized-damage
primary analysis.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis import (
    AnalysisValidationError,
    exact_paired_sign_flip,
    hierarchical_bootstrap,
    load_rows,
    validate_expected_keys,
    validate_paired_edit_hashes,
)

FLOOR_GRID = (0.5000000001, 0.525, 0.55, 0.575, 0.60)

_SCHEMA = "reviewer_revision.floor_robustness.v1"
_PAIR_COLUMNS = ("model_key", "task", "layer", "pair_seed")
_CELL_COLUMNS = ("model_key", "task", "layer")
_BLOCK_COLUMNS = ("model_key", "task")
_KEY_COLUMNS = (*_PAIR_COLUMNS, "method", "condition")
_CONDITIONS = ("matched", "split")
_SCORE_NULL_STATUSES = frozenset(("pre_target_below_floor", "pre_control_below_floor"))
_HARD_FAILURE_STATUSES = frozenset(("failed", "invalid"))
_ANALYZABLE_STATUSES = frozenset(("ok", *_SCORE_NULL_STATUSES))
_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "model_key": ("model_key", "model"),
    "task": ("task", "task_key"),
    "layer": ("layer",),
    "pair_seed": ("pair_seed",),
    "method": ("method",),
    "condition": ("condition", "scoring_condition"),
    "target_acc_pre": (
        "target_acc_pre",
        "target_accuracy_pre",
        "raw_target_accuracy_pre",
    ),
    "target_acc_post": (
        "target_acc_post",
        "target_accuracy_post",
        "raw_target_accuracy_post",
    ),
    "chance": ("chance",),
    "status": ("status",),
    "edit_hash": ("edit_hash",),
}


def _resolved_column(frame: pd.DataFrame, canonical: str) -> str:
    for candidate in _COLUMN_CANDIDATES[canonical]:
        if candidate in frame.columns:
            return candidate
    raise AnalysisValidationError(
        f"row artifact is missing required column {canonical!r}; "
        f"accepted names are {list(_COLUMN_CANDIDATES[canonical])!r}"
    )


def _numeric_measure(
    frame: pd.DataFrame,
    canonical: str,
    *,
    require_unit_interval: bool = True,
) -> pd.Series:
    source = _resolved_column(frame, canonical)
    values = pd.to_numeric(frame[source], errors="coerce").astype(float)
    finite = np.isfinite(values.to_numpy(dtype=float))
    if not finite.all():
        bad = frame.index[~finite].tolist()
        raise AnalysisValidationError(
            f"non-finite values in {canonical!r} at rows {bad[:10]}"
        )
    if require_unit_interval and ((values < 0.0) | (values > 1.0)).any():
        bad = frame.index[((values < 0.0) | (values > 1.0))].tolist()
        raise AnalysisValidationError(
            f"{canonical!r} must lie in [0, 1]; invalid rows {bad[:10]}"
        )
    return values


def raw_target_drop(frame: pd.DataFrame) -> pd.Series:
    """Return the unnormalized target-accuracy drop for every row."""

    pre = _numeric_measure(frame, "target_acc_pre")
    post = _numeric_measure(frame, "target_acc_post")
    return (pre - post).rename("raw_target_drop")


def target_damage_at_floor(frame: pd.DataFrame, floor: float) -> pd.Series:
    """Recompute clipped target-only normalized damage at an exact floor.

    Rows whose pre-edit target accuracy is strictly below ``floor`` are null.
    The comparison is deliberately ordinary ``>=`` rather than rounded or
    approximate so threshold-boundary behavior is auditable.
    """

    try:
        numeric_floor = float(floor)
    except (TypeError, ValueError) as exc:
        raise AnalysisValidationError("floor must be finite and numeric") from exc
    if not np.isfinite(numeric_floor) or not 0.5 < numeric_floor <= 1.0:
        raise AnalysisValidationError("floor must be finite and in (0.5, 1]")
    pre = _numeric_measure(frame, "target_acc_pre")
    post = _numeric_measure(frame, "target_acc_post")
    chance = _numeric_measure(frame, "chance")
    if (chance != 0.5).any():
        raise AnalysisValidationError(
            "chance must be exactly 0.5 using ordinary comparison"
        )
    eligible = pre >= numeric_floor
    denominator = pre - chance
    if (denominator.loc[eligible] <= 0.0).any():
        raise AnalysisValidationError(
            "eligible target-damage rows require target_acc_pre above chance"
        )
    damage = pd.Series(np.nan, index=frame.index, dtype=float)
    damage.loc[eligible] = (
        (pre.loc[eligible] - post.loc[eligible]) / denominator.loc[eligible]
    ).clip(0.0, 1.0)
    return damage.rename("target_damage_at_floor")


def _source_provenance(
    source: pd.DataFrame | Path | str | Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    if isinstance(source, (str, os.PathLike)):
        path = Path(source).resolve()
        if not path.is_file():
            raise AnalysisValidationError(f"row artifact does not exist: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "input_kind": "path",
            "source_path": str(path),
            "source_sha256": digest.hexdigest(),
        }
    return {
        "input_kind": ("dataframe" if isinstance(source, pd.DataFrame) else "iterable"),
        "source_path": None,
        "source_sha256": None,
    }


def _nonblank(value: Any) -> bool:
    return not pd.isna(value) and bool(str(value).strip())


def _finite_nonblank_key(value: Any) -> bool:
    if not _nonblank(value):
        return False
    if isinstance(value, (int, float, np.number)):
        return bool(np.isfinite(value))
    return True


def _canonical_alterrep_rows(
    source: pd.DataFrame | Path | str | Iterable[Mapping[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = load_rows(source)
    aliases = {
        canonical: _resolved_column(raw, canonical) for canonical in _COLUMN_CANDIDATES
    }
    canonical = pd.DataFrame(
        {name: raw[source_name] for name, source_name in aliases.items()}
    )
    selected = canonical.loc[canonical["method"] == "alterrep"].copy()
    if selected.empty:
        raise AnalysisValidationError("no rows found for method 'alterrep'")

    for column in ("model_key", "task", "method", "condition", "status"):
        validator = (
            _finite_nonblank_key if column in ("model_key", "task") else _nonblank
        )
        invalid = ~selected[column].map(validator)
        if invalid.any():
            raise AnalysisValidationError(
                f"{column!r} contains null or blank keys at rows "
                f"{selected.index[invalid].tolist()[:10]}"
            )
        selected[column] = selected[column].map(str)

    invalid_conditions = ~selected["condition"].isin(_CONDITIONS)
    if invalid_conditions.any():
        observed = sorted(selected.loc[invalid_conditions, "condition"].unique())
        raise AnalysisValidationError(
            "alterrep condition must be exactly 'matched' or 'split'; "
            f"observed {observed!r}"
        )

    hard_failures = selected["status"].isin(_HARD_FAILURE_STATUSES)
    if hard_failures.any():
        counts = {
            status: int(count)
            for status, count in selected.loc[hard_failures, "status"]
            .value_counts()
            .sort_index()
            .items()
        }
        raise AnalysisValidationError(
            f"alterrep rows contain hard-failure statuses: {counts!r}"
        )
    unsupported = ~selected["status"].isin(_ANALYZABLE_STATUSES)
    if unsupported.any():
        observed = sorted(selected.loc[unsupported, "status"].unique())
        raise AnalysisValidationError(
            f"alterrep rows contain unsupported status values: {observed!r}"
        )

    for column in ("layer", "pair_seed"):
        numeric = pd.to_numeric(selected[column], errors="coerce").astype(float)
        finite = np.isfinite(numeric.to_numpy(dtype=float))
        if not finite.all():
            bad = selected.index[~finite].tolist()
            raise AnalysisValidationError(
                f"non-finite values in {column!r} at rows {bad[:10]}"
            )
        nonintegral = numeric != np.floor(numeric)
        if nonintegral.any():
            bad = selected.index[nonintegral].tolist()
            raise AnalysisValidationError(
                f"{column!r} keys must be integer-valued at rows {bad[:10]}"
            )
        selected[column] = numeric.astype(int)

    selected["target_acc_pre"] = _numeric_measure(selected, "target_acc_pre")
    selected["target_acc_post"] = _numeric_measure(selected, "target_acc_post")
    selected["chance"] = _numeric_measure(selected, "chance")
    if (selected["chance"] != 0.5).any():
        bad = selected.index[selected["chance"] != 0.5].tolist()
        raise AnalysisValidationError(
            "chance must be exactly 0.5 using ordinary comparison; "
            f"invalid rows {bad[:10]}"
        )

    invalid_hash = ~selected["edit_hash"].map(_nonblank)
    if invalid_hash.any():
        raise AnalysisValidationError(
            "edit_hash contains null or blank values at rows "
            f"{selected.index[invalid_hash].tolist()[:10]}"
        )
    selected["edit_hash"] = selected["edit_hash"].map(str)
    selected = selected.sort_values(
        [*_PAIR_COLUMNS, "condition"], kind="mergesort"
    ).reset_index(drop=True)

    duplicates = selected.duplicated([*_PAIR_COLUMNS, "condition"], keep=False)
    if duplicates.any():
        raise AnalysisValidationError(
            "alterrep rows contain duplicate pair-condition keys"
        )
    expected_keys: list[tuple[Any, ...]] = []
    for raw_pair, group in selected.groupby(
        list(_PAIR_COLUMNS), sort=True, dropna=False
    ):
        pair = raw_pair if isinstance(raw_pair, tuple) else (raw_pair,)
        conditions = tuple(sorted(group["condition"].tolist()))
        if len(group) != 2 or conditions != _CONDITIONS:
            raise AnalysisValidationError(
                "every alterrep pair requires exactly matched and split rows; "
                f"invalid pair {pair!r} has conditions {conditions!r}"
            )
        expected_keys.extend(
            (*pair, "alterrep", condition) for condition in _CONDITIONS
        )

    completeness = validate_expected_keys(
        selected,
        expected_keys,
        _KEY_COLUMNS,
        success_statuses=tuple(sorted(_ANALYZABLE_STATUSES)),
        failure_statuses=(),
    )
    validated_pairs = validate_paired_edit_hashes(
        selected,
        identity_columns=_PAIR_COLUMNS,
        condition_column="condition",
        edit_hash_column="edit_hash",
    )
    status_counts = {
        status: int(count)
        for status, count in selected["status"].value_counts().sort_index().items()
    }
    validation = {
        "input_rows": len(raw),
        "selected_rows": len(selected),
        "selected_method": "alterrep",
        "required_conditions": list(_CONDITIONS),
        "chance": 0.5,
        "status_counts": status_counts,
        "score_null_rows": int(selected["status"].isin(_SCORE_NULL_STATUSES).sum()),
        "hard_failure_rows": 0,
        "validated_paired_edit_hashes": int(validated_pairs),
        "expected_key_count": int(completeness.expected_count),
        "observed_key_count": int(completeness.observed_count),
        "keys_complete": bool(completeness.is_complete),
        "column_aliases": aliases,
    }
    return selected, validation


def _paired_values(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    paired = (
        frame.pivot(
            index=list(_PAIR_COLUMNS),
            columns="condition",
            values=value_column,
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    paired["gap"] = paired["matched"] - paired["split"]
    return paired.sort_values(list(_PAIR_COLUMNS), kind="mergesort").reset_index(
        drop=True
    )


def _hierarchical_means(
    paired: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    cells = (
        paired.groupby(list(_CELL_COLUMNS), sort=True, as_index=False)[
            ["matched", "split", "gap"]
        ]
        .mean()
        .sort_values(list(_CELL_COLUMNS), kind="mergesort")
        .reset_index(drop=True)
    )
    blocks = (
        cells.groupby(list(_BLOCK_COLUMNS), sort=True, as_index=False)[
            ["matched", "split", "gap"]
        ]
        .mean()
        .sort_values(list(_BLOCK_COLUMNS), kind="mergesort")
        .reset_index(drop=True)
    )
    cells_per_block = cells.groupby(list(_BLOCK_COLUMNS), sort=True).size()
    # With the complete planned hierarchy, every block has the same number of
    # selected cells, so the direct cell mean is exactly the equal-block
    # estimand and preserves the registered audit's floating-point reduction
    # order.  Once whole cells disappear, use explicit block means so surviving
    # cells in one block cannot outweigh another block.
    point_frame = cells if cells_per_block.nunique() == 1 else blocks
    means: dict[str, float | int] = {
        "matched_mean": float(point_frame["matched"].mean()),
        "split_mean": float(point_frame["split"].mean()),
        "gap": float(point_frame["gap"].mean()),
        "pairs": len(paired),
        "cells": len(cells),
        "model_task_blocks": len(blocks),
    }
    return cells, blocks, means


def _sign_counts(values: pd.Series) -> dict[str, int]:
    array = values.to_numpy(dtype=float)
    return {
        "positive": int(np.count_nonzero(array > 0.0)),
        "zero": int(np.count_nonzero(array == 0.0)),
        "negative": int(np.count_nonzero(array < 0.0)),
    }


def _pair_key_list(key: tuple[Any, ...]) -> list[Any]:
    return [str(key[0]), str(key[1]), int(key[2]), int(key[3])]


def _cell_key_list(key: tuple[Any, ...]) -> list[Any]:
    return [str(key[0]), str(key[1]), int(key[2])]


def _floor_pair_data(
    alterrep: pd.DataFrame, floor: float
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    work = alterrep.copy()
    work["floor_damage"] = target_damage_at_floor(work, floor)
    complete = work.groupby(list(_PAIR_COLUMNS), sort=False, dropna=False)[
        "floor_damage"
    ].transform(lambda values: bool(values.notna().all()))
    analyzed = work.loc[complete].copy()
    paired = (
        _paired_values(analyzed, "floor_damage")
        if not analyzed.empty
        else pd.DataFrame(columns=[*_PAIR_COLUMNS, "matched", "split", "gap"])
    )
    excluded: list[dict[str, Any]] = []
    for raw_key, group in work.groupby(list(_PAIR_COLUMNS), sort=True, dropna=False):
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        reasons = [
            f"{row.condition}_target_acc_pre_below_floor"
            for row in group.sort_values("condition").itertuples(index=False)
            if float(row.target_acc_pre) < floor
        ]
        if reasons:
            excluded.append(
                {
                    "pair_key": _pair_key_list(tuple(key)),
                    "reasons": reasons,
                }
            )
    return paired, excluded


def _coverage_at_floor(alterrep: pd.DataFrame, paired: pd.DataFrame) -> dict[str, Any]:
    planned_pairs = alterrep[list(_PAIR_COLUMNS)].drop_duplicates()
    planned_counts = planned_pairs.groupby(
        list(_CELL_COLUMNS), sort=True, dropna=False
    ).size()
    if paired.empty:
        observed_counts: dict[tuple[Any, ...], int] = {}
    else:
        observed_counts = {
            tuple(key) if isinstance(key, tuple) else (key,): int(count)
            for key, count in paired.groupby(
                list(_CELL_COLUMNS), sort=True, dropna=False
            )
            .size()
            .items()
        }
    full: list[list[Any]] = []
    partial: list[list[Any]] = []
    missing: list[list[Any]] = []
    for raw_key, planned in planned_counts.items():
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        observed = observed_counts.get(tuple(key), 0)
        serialized = _cell_key_list(tuple(key))
        if observed == int(planned):
            full.append(serialized)
        elif observed == 0:
            missing.append(serialized)
        else:
            partial.append(serialized)
    return {
        "requested_cells": len(planned_counts),
        "full_cells": len(full),
        "partial_cells": len(partial),
        "missing_cells": len(missing),
        "full_cell_keys": full,
        "partial_cell_keys": partial,
        "missing_cell_keys": missing,
    }


def _not_estimable_sign_flip() -> dict[str, Any]:
    return {
        "status": "not_estimable",
        "method": "two_sided_exact_paired_sign_flip",
        "reason": "no model-task block has an analyzed pair at this floor",
        "n_blocks": 0,
        "nonzero_count": 0,
        "zero_count": 0,
        "p_value": None,
    }


def _floor_summary(alterrep: pd.DataFrame, floor: float) -> dict[str, Any]:
    paired, excluded = _floor_pair_data(alterrep, floor)
    coverage = _coverage_at_floor(alterrep, paired)
    requested_pairs = int(alterrep[list(_PAIR_COLUMNS)].drop_duplicates().shape[0])
    requested_blocks = int(alterrep[list(_BLOCK_COLUMNS)].drop_duplicates().shape[0])
    if paired.empty:
        blocks = pd.DataFrame(columns=[*_BLOCK_COLUMNS, "matched", "split", "gap"])
        means: dict[str, Any] = {
            "matched_mean": None,
            "split_mean": None,
            "gap": None,
            "pairs": 0,
            "cells": 0,
            "model_task_blocks": 0,
        }
        sign_flip = _not_estimable_sign_flip()
        block_signs = {"positive": 0, "zero": 0, "negative": 0}
    else:
        _cells, blocks, means = _hierarchical_means(paired)
        sign_flip = exact_paired_sign_flip(
            blocks["gap"].to_numpy(), expected_blocks=len(blocks)
        )
        sign_flip["status"] = "ok"
        block_signs = _sign_counts(blocks["gap"])
    equal_block = {
        "matched": means["matched_mean"],
        "split": means["split_mean"],
        "gap": means["gap"],
    }
    return {
        "floor": float(floor),
        "label": "post_hoc_sensitivity",
        "estimand": (
            "Available-case target-only normalized damage after symmetric pair "
            "exclusion at the stated exact pre-edit target-accuracy floor; "
            "pair means within cells, selected estimable cells within model-task "
            "blocks, and equal weight across estimable blocks."
        ),
        "comparison": "matched minus split on the identical edit",
        "requested_pairs": requested_pairs,
        "analyzed_pairs": int(means["pairs"]),
        "pairs": int(means["pairs"]),
        "analyzed_cells": int(means["cells"]),
        "requested_model_task_blocks": requested_blocks,
        "analyzed_model_task_blocks": int(means["model_task_blocks"]),
        **coverage,
        "excluded_pair_keys": [row["pair_key"] for row in excluded],
        "excluded_pairs": excluded,
        "matched_mean": means["matched_mean"],
        "split_mean": means["split_mean"],
        "gap": means["gap"],
        "equal_block": equal_block,
        "exact_sign_flip": sign_flip,
        "block_gap_sign_counts": block_signs,
        "hierarchical_bootstrap_reported": False,
        "hierarchical_bootstrap_omission_reason": (
            "Whole cells can disappear at higher floors; the existing bootstrap "
            "helper reports an equal-cell point estimate, not this equal-block "
            "available-case point estimate."
        ),
    }


def _partial_identification(
    alterrep: pd.DataFrame, available_case_gap: float | None
) -> dict[str, Any]:
    paired, _ = _floor_pair_data(alterrep, 0.55)
    planned_pairs = alterrep[list(_PAIR_COLUMNS)].drop_duplicates()
    planned_counts = planned_pairs.groupby(
        list(_CELL_COLUMNS), sort=True, dropna=False
    ).size()
    observed_sums: dict[tuple[Any, ...], float] = {}
    observed_counts: dict[tuple[Any, ...], int] = {}
    if not paired.empty:
        for raw_key, group in paired.groupby(
            list(_CELL_COLUMNS), sort=True, dropna=False
        ):
            key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
            observed_sums[tuple(key)] = float(group["gap"].sum())
            observed_counts[tuple(key)] = len(group)

    cell_rows: list[dict[str, Any]] = []
    for raw_key, raw_planned in planned_counts.items():
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        planned = int(raw_planned)
        observed = observed_counts.get(tuple(key), 0)
        missing = planned - observed
        observed_sum = observed_sums.get(tuple(key), 0.0)
        cell_rows.append(
            {
                "model_key": str(key[0]),
                "task": str(key[1]),
                "layer": int(key[2]),
                "planned_pairs": planned,
                "observed_pairs": observed,
                "missing_pairs": missing,
                "lower": float((observed_sum - missing) / planned),
                "upper": float((observed_sum + missing) / planned),
            }
        )
    cells = pd.DataFrame(cell_rows)
    blocks = (
        cells.groupby(list(_BLOCK_COLUMNS), sort=True, as_index=False)[
            ["lower", "upper"]
        ]
        .mean()
        .sort_values(list(_BLOCK_COLUMNS), kind="mergesort")
        .reset_index(drop=True)
    )
    observed_pairs = len(paired)
    planned_pair_count = len(planned_pairs)
    return {
        "floor": 0.55,
        "label": "deterministic_partial_identification_bound",
        "sensitivity_label": "post_hoc_sensitivity",
        "estimand": (
            "Worst-case full-plan equal-block gap: bound each missing paired "
            "normalized-damage gap in [-1, 1], divide observed gap sums plus "
            "or minus missing counts by every planned cell's pair count, average "
            "all planned cells within fixed model-task blocks, then blocks equally."
        ),
        "comparison": "matched minus split on the identical edit",
        "is_confidence_interval": False,
        "available_case_gap": available_case_gap,
        "missing_pair_gap_domain": [-1.0, 1.0],
        "planned_pairs": planned_pair_count,
        "observed_pairs": observed_pairs,
        "missing_pairs": planned_pair_count - observed_pairs,
        "planned_cells": len(cells),
        "wholly_missing_cells": int((cells["observed_pairs"] == 0).sum()),
        "fixed_model_task_blocks": len(blocks),
        "lower": float(blocks["lower"].mean()),
        "upper": float(blocks["upper"].mean()),
        "cell_bounds": cell_rows,
    }


def summarize_floor_robustness(
    rows: pd.DataFrame | Path | str | Iterable[Mapping[str, Any]],
    *,
    draws: int = 10_000,
    seed: int = 20260830,
) -> dict[str, Any]:
    """Return deterministic post-hoc raw-drop and floor-robustness analyses."""

    provenance = _source_provenance(rows)
    alterrep, validation = _canonical_alterrep_rows(rows)
    alterrep["raw_drop"] = raw_target_drop(alterrep)
    raw_pairs = _paired_values(alterrep, "raw_drop")
    raw_cells, raw_blocks, raw_means = _hierarchical_means(raw_pairs)
    bootstrap_frame = raw_pairs.rename(columns={"model_key": "model"})
    bootstrap = hierarchical_bootstrap(
        bootstrap_frame,
        value_column="gap",
        block_columns=("model", "task"),
        layer_column="layer",
        pair_column="pair_seed",
        draws=draws,
        seed=seed,
    )
    bootstrap["equal_block_point_estimate"] = raw_means["gap"]
    raw_sign_flip = exact_paired_sign_flip(
        raw_blocks["gap"].to_numpy(), expected_blocks=len(raw_blocks)
    )
    raw_sign_flip["status"] = "ok"
    raw_summary = {
        **raw_means,
        "label": "post_hoc_sensitivity",
        "post_hoc_sensitivity": True,
        "replaces_registered_primary": False,
        "estimand": (
            "Full-case raw target-accuracy drop; pair rows within cells, selected "
            "cells within model-task blocks, and equal weight across blocks."
        ),
        "comparison": (
            "(target_acc_pre - target_acc_post) matched minus split on the "
            "identical edit"
        ),
        "bootstrap": bootstrap,
        "exact_sign_flip": raw_sign_flip,
        "block_gap_sign_counts": _sign_counts(raw_blocks["gap"]),
    }
    floor_curve = [_floor_summary(alterrep, floor) for floor in FLOOR_GRID]
    locked_floor = next(row for row in floor_curve if row["floor"] == 0.55)
    partial = _partial_identification(alterrep, locked_floor["gap"])
    units = {
        "rows": len(alterrep),
        "pairs": len(raw_pairs),
        "cells": len(raw_cells),
        "model_task_blocks": len(raw_blocks),
    }
    return {
        "schema": _SCHEMA,
        "schema_version": 1,
        "status": "ok",
        "label": "post_hoc_sensitivity",
        "post_hoc_sensitivity": True,
        "replaces_registered_primary": False,
        "estimand": (
            "Post-hoc denominator-floor robustness of the matched-versus-split "
            "AlterRep contrast under equal model-task block weighting."
        ),
        "comparison": "matched minus split on the identical edit",
        "registered_primary_relation": (
            "Sensitivity analysis only; does not replace or relabel the registered "
            "normalized-damage primary analysis."
        ),
        "provenance": provenance,
        "validation": validation,
        "units": units,
        "full_case_raw_drop": raw_summary,
        "raw_drop": raw_summary,
        "floor_curve": floor_curve,
        "partial_identification": partial,
    }


__all__ = [
    "FLOOR_GRID",
    "raw_target_drop",
    "summarize_floor_robustness",
    "target_damage_at_floor",
]
