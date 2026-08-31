"""Post-hoc denominator-floor robustness analyses for the reviewer revision.

These summaries are derived only from materialized matched/split rows.  They
are sensitivity analyses and never replace the registered normalized-damage
primary analysis.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis import (
    AnalysisValidationError,
    exact_paired_sign_flip,
    load_rows,
    validate_expected_keys,
    validate_paired_edit_hashes,
)
from .extension_config import ConstructCell, ExtensionConfig

FLOOR_GRID = (0.5000000001, 0.525, 0.55, 0.575, 0.60)

LOCKED_CONSTRUCT_CANDIDATE_EDIT_IDS = tuple(
    f"dcand_crossfit-{architecture}-seed{seed}"
    for architecture in ("linear", "mlp", "mka")
    for seed in range(20)
)

_SCHEMA = "reviewer_revision.floor_robustness.v1"
_LOCKED_EXPECTED_PAIRS = 300
_LOCKED_EXPECTED_CELLS = 60
_LOCKED_EXPECTED_BLOCKS = 12
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


def _require_unique_column_labels(frame: pd.DataFrame) -> None:
    duplicates = frame.columns[frame.columns.duplicated(keep=False)].tolist()
    if duplicates:
        ordered = list(dict.fromkeys(duplicates))
        raise AnalysisValidationError(
            f"row artifact contains duplicate column labels: {ordered!r}"
        )


def _alias_values_equivalent(left: pd.Series, right: pd.Series) -> bool:
    left_missing = left.isna()
    right_missing = right.isna()
    if not left_missing.equals(right_missing):
        return False
    populated = ~left_missing
    if not populated.any():
        return True
    try:
        left_values = left.loc[populated].astype(object).reset_index(drop=True)
        right_values = right.loc[populated].astype(object).reset_index(drop=True)
    except (TypeError, ValueError):
        return False
    return left_values.equals(right_values)


def _resolved_column(frame: pd.DataFrame, canonical: str) -> str:
    _require_unique_column_labels(frame)
    present = [
        candidate
        for candidate in _COLUMN_CANDIDATES[canonical]
        if candidate in frame.columns
    ]
    if not present:
        raise AnalysisValidationError(
            f"row artifact is missing required column {canonical!r}; "
            f"accepted names are {list(_COLUMN_CANDIDATES[canonical])!r}"
        )
    populated = [candidate for candidate in present if frame[candidate].notna().any()]
    selected = populated[0] if populated else present[0]
    for alias in populated[1:]:
        if not _alias_values_equivalent(frame[selected], frame[alias]):
            raise AnalysisValidationError(
                f"conflicting populated aliases for {canonical!r}: "
                f"{selected!r} and {alias!r}"
            )
    return selected


def _numeric_measure(
    frame: pd.DataFrame,
    canonical: str,
    *,
    require_unit_interval: bool = True,
) -> pd.Series:
    source = _resolved_column(frame, canonical)
    boolean = frame[source].map(lambda value: isinstance(value, (bool, np.bool_)))
    if boolean.any():
        bad = frame.index[boolean].tolist()
        raise AnalysisValidationError(
            f"{canonical!r} must not contain boolean values; invalid rows {bad[:10]}"
        )
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

    if isinstance(floor, (bool, np.bool_)):
        raise AnalysisValidationError("floor must be numeric, not boolean")
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
            "source_name": path.name,
            "source_sha256": digest.hexdigest(),
        }
    return {
        "input_kind": ("dataframe" if isinstance(source, pd.DataFrame) else "iterable"),
        "source_name": None,
        "source_sha256": None,
    }


def _nonblank_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _integer_scalar(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _validated_expected_count(value: Any, name: str) -> int:
    if not _integer_scalar(value) or value <= 0:
        raise AnalysisValidationError(f"{name} must be a positive non-boolean integer")
    return int(value)


def _validated_seed(value: Any) -> int:
    if not _integer_scalar(value) or value < 0:
        raise AnalysisValidationError("seed must be a nonnegative non-boolean integer")
    return int(value)


def _strict_expected_key(item: Any) -> tuple[Any, ...]:
    if isinstance(item, Mapping):
        missing = [column for column in _KEY_COLUMNS if column not in item]
        if missing:
            raise AnalysisValidationError(
                f"expected AlterRep key mapping is missing columns: {missing!r}"
            )
        key = tuple(item[column] for column in _KEY_COLUMNS)
    else:
        try:
            key = tuple(item)
        except TypeError as exc:
            raise AnalysisValidationError(
                f"expected AlterRep key {item!r} is not iterable"
            ) from exc
        if len(key) != len(_KEY_COLUMNS):
            raise AnalysisValidationError(
                f"expected AlterRep key {item!r} has {len(key)} fields; "
                f"expected {len(_KEY_COLUMNS)}"
            )
    for column, index in (
        ("model_key", 0),
        ("task", 1),
        ("method", 4),
        ("condition", 5),
    ):
        if not _nonblank_string(key[index]):
            raise AnalysisValidationError(
                f"expected AlterRep key {column!r} must be a nonblank string"
            )
    if not _integer_scalar(key[2]) or key[2] <= 0:
        raise AnalysisValidationError(
            "expected AlterRep key 'layer' must be a positive non-boolean integer"
        )
    if not _integer_scalar(key[3]) or key[3] < 0:
        raise AnalysisValidationError(
            "expected AlterRep key 'pair_seed' must be a nonnegative "
            "non-boolean integer"
        )
    if key[4] != "alterrep":
        raise AnalysisValidationError(
            "expected_alterrep_keys may contain only method 'alterrep'"
        )
    if key[5] not in _CONDITIONS:
        raise AnalysisValidationError(
            "expected_alterrep_keys conditions must be exactly matched or split"
        )
    return key


def _hierarchy_shape(frame: pd.DataFrame, *, context: str) -> dict[str, Any]:
    pairs = frame[list(_PAIR_COLUMNS)].drop_duplicates()
    cells = pairs[list(_CELL_COLUMNS)].drop_duplicates()
    pairs_per_cell = pairs.groupby(list(_CELL_COLUMNS), sort=True, dropna=False).size()
    cells_per_block = cells.groupby(
        list(_BLOCK_COLUMNS), sort=True, dropna=False
    ).size()
    if pairs_per_cell.empty or (pairs_per_cell <= 0).any():
        raise AnalysisValidationError(
            f"{context} must contain a positive number of pairs per cell"
        )
    if cells_per_block.empty or (cells_per_block <= 0).any():
        raise AnalysisValidationError(
            f"{context} must contain a positive number of cells per model-task block"
        )
    pair_counts = sorted({int(value) for value in pairs_per_cell.tolist()})
    cell_counts = sorted({int(value) for value in cells_per_block.tolist()})
    return {
        "balanced": len(pair_counts) == 1 and len(cell_counts) == 1,
        "pairs_per_cell": pair_counts[0] if len(pair_counts) == 1 else None,
        "cells_per_model_task_block": (
            cell_counts[0] if len(cell_counts) == 1 else None
        ),
        "pairs_per_cell_values": pair_counts,
        "cells_per_model_task_block_values": cell_counts,
    }


def _validate_expected_universe_structure(
    expected_keys: Sequence[tuple[Any, ...]],
    *,
    expected_pairs: int,
    expected_cells: int,
    expected_blocks: int,
) -> dict[str, Any]:
    planned = pd.DataFrame(expected_keys, columns=_KEY_COLUMNS)
    for raw_pair, group in planned.groupby(
        list(_PAIR_COLUMNS), sort=True, dropna=False
    ):
        conditions = tuple(sorted(group["condition"].tolist()))
        if len(group) != 2 or conditions != _CONDITIONS:
            raise AnalysisValidationError(
                "expected_alterrep_keys requires exactly matched and split for "
                f"planned pair {raw_pair!r}; observed {conditions!r}"
            )
    counts = {
        "pairs": len(planned[list(_PAIR_COLUMNS)].drop_duplicates()),
        "cells": len(planned[list(_CELL_COLUMNS)].drop_duplicates()),
        "model-task blocks": len(planned[list(_BLOCK_COLUMNS)].drop_duplicates()),
    }
    expected = {
        "pairs": expected_pairs,
        "cells": expected_cells,
        "model-task blocks": expected_blocks,
    }
    for unit, expected_count in expected.items():
        if counts[unit] != expected_count:
            raise AnalysisValidationError(
                "external expected AlterRep key universe conflicts with expected "
                f"counts: expected {expected_count} {unit}, universe defines "
                f"{counts[unit]}"
            )
    return _hierarchy_shape(planned, context="external expected AlterRep key universe")


def _canonical_alterrep_rows(
    source: pd.DataFrame | Path | str | Iterable[Mapping[str, Any]],
    *,
    expected_alterrep_keys: Iterable[Any] | None,
    expected_pairs: int,
    expected_cells: int,
    expected_blocks: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    expected_pairs = _validated_expected_count(expected_pairs, "expected_pairs")
    expected_cells = _validated_expected_count(expected_cells, "expected_cells")
    expected_blocks = _validated_expected_count(expected_blocks, "expected_blocks")
    raw = load_rows(source)
    method_source = _resolved_column(raw, "method")
    invalid_methods = ~raw[method_source].map(_nonblank_string)
    if invalid_methods.any():
        raise AnalysisValidationError(
            "'method' must contain actual nonblank strings; invalid rows "
            f"{raw.index[invalid_methods].tolist()[:10]}"
        )
    selected_raw = raw.loc[raw[method_source] == "alterrep"].copy()
    if selected_raw.empty:
        raise AnalysisValidationError("no rows found for method 'alterrep'")
    aliases = {
        canonical: _resolved_column(selected_raw, canonical)
        for canonical in _COLUMN_CANDIDATES
    }
    selected = pd.DataFrame(
        {name: selected_raw[source_name] for name, source_name in aliases.items()}
    )

    for column in (
        "model_key",
        "task",
        "method",
        "condition",
        "status",
        "edit_hash",
    ):
        invalid = ~selected[column].map(_nonblank_string)
        if invalid.any():
            raise AnalysisValidationError(
                f"{column!r} must contain actual nonblank strings; invalid rows "
                f"{selected.index[invalid].tolist()[:10]}"
            )

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

    valid_layers = selected["layer"].map(
        lambda value: _integer_scalar(value) and value > 0
    )
    if not valid_layers.all():
        raise AnalysisValidationError(
            "'layer' must contain positive non-boolean integer values; invalid rows "
            f"{selected.index[~valid_layers].tolist()[:10]}"
        )
    valid_pair_seeds = selected["pair_seed"].map(
        lambda value: _integer_scalar(value) and value >= 0
    )
    if not valid_pair_seeds.all():
        raise AnalysisValidationError(
            "'pair_seed' must contain nonnegative non-boolean integer values; "
            f"invalid rows {selected.index[~valid_pair_seeds].tolist()[:10]}"
        )

    selected["target_acc_pre"] = _numeric_measure(selected, "target_acc_pre")
    selected["target_acc_post"] = _numeric_measure(selected, "target_acc_post")
    selected["chance"] = _numeric_measure(selected, "chance")
    if (selected["chance"] != 0.5).any():
        bad = selected.index[selected["chance"] != 0.5].tolist()
        raise AnalysisValidationError(
            "chance must be exactly 0.5 using ordinary comparison; "
            f"invalid rows {bad[:10]}"
        )

    selected = selected.sort_values(
        [*_PAIR_COLUMNS, "condition"], kind="mergesort"
    ).reset_index(drop=True)

    duplicates = selected.duplicated([*_PAIR_COLUMNS, "condition"], keep=False)
    if duplicates.any():
        raise AnalysisValidationError(
            "alterrep rows contain duplicate pair-condition keys"
        )
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

    completeness = None
    normalized_expected_keys: tuple[tuple[Any, ...], ...] | None = None
    planned_shape: dict[str, Any] | None = None
    if expected_alterrep_keys is not None:
        normalized_expected_keys = tuple(
            _strict_expected_key(item) for item in expected_alterrep_keys
        )
        if not normalized_expected_keys:
            raise AnalysisValidationError("expected_alterrep_keys must not be empty")
        completeness = validate_expected_keys(
            selected,
            normalized_expected_keys,
            _KEY_COLUMNS,
            success_statuses=tuple(sorted(_ANALYZABLE_STATUSES)),
            failure_statuses=(),
        )
        planned_shape = _validate_expected_universe_structure(
            normalized_expected_keys,
            expected_pairs=expected_pairs,
            expected_cells=expected_cells,
            expected_blocks=expected_blocks,
        )

    observed_counts = {
        "pairs": len(selected[list(_PAIR_COLUMNS)].drop_duplicates()),
        "cells": len(selected[list(_CELL_COLUMNS)].drop_duplicates()),
        "model-task blocks": len(selected[list(_BLOCK_COLUMNS)].drop_duplicates()),
    }
    required_counts = {
        "pairs": expected_pairs,
        "cells": expected_cells,
        "model-task blocks": expected_blocks,
    }
    for unit, expected_count in required_counts.items():
        if observed_counts[unit] != expected_count:
            raise AnalysisValidationError(
                "alterrep planned-count mismatch: expected "
                f"{expected_count} {unit}, observed {observed_counts[unit]}"
            )
    observed_shape = _hierarchy_shape(
        selected, context="observed AlterRep row universe"
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
    locked_counts = (
        expected_pairs,
        expected_cells,
        expected_blocks,
    ) == (
        _LOCKED_EXPECTED_PAIRS,
        _LOCKED_EXPECTED_CELLS,
        _LOCKED_EXPECTED_BLOCKS,
    )
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
        "completeness_mode": (
            "external_key_universe"
            if completeness is not None
            else "expected_counts_only"
        ),
        "external_key_universe_checked": completeness is not None,
        "observed_derived_key_universe": False,
        "expected_count_source": (
            "locked_defaults" if locked_counts else "explicit_overrides"
        ),
        "expected_counts": {
            "pairs": expected_pairs,
            "cells": expected_cells,
            "model_task_blocks": expected_blocks,
        },
        "observed_counts": {
            "pairs": observed_counts["pairs"],
            "cells": observed_counts["cells"],
            "model_task_blocks": observed_counts["model-task blocks"],
        },
        "balanced_hierarchy_checked": True,
        "balanced_hierarchy": observed_shape["balanced"],
        "pairs_per_cell": observed_shape["pairs_per_cell"],
        "cells_per_model_task_block": observed_shape["cells_per_model_task_block"],
        "pairs_per_cell_values": observed_shape["pairs_per_cell_values"],
        "cells_per_model_task_block_values": observed_shape[
            "cells_per_model_task_block_values"
        ],
        "hierarchy_balance": {
            "required": False,
            "observed": observed_shape,
            "external_planned": planned_shape,
        },
        "expected_key_count": (
            completeness.expected_count if completeness is not None else None
        ),
        "observed_key_count": len(selected),
        "keys_complete": (
            bool(completeness.is_complete) if completeness is not None else None
        ),
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


def _equal_block_hierarchical_bootstrap(
    paired: pd.DataFrame,
    *,
    draws: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap pairs, cells, and model-task blocks with equal block weight."""

    ordered = paired.sort_values(list(_PAIR_COLUMNS), kind="mergesort")
    hierarchy: list[tuple[tuple[Any, ...], list[np.ndarray]]] = []
    for raw_block, block_frame in ordered.groupby(
        list(_BLOCK_COLUMNS), sort=True, dropna=False
    ):
        block = raw_block if isinstance(raw_block, tuple) else (raw_block,)
        cells = [
            cell_frame["gap"].to_numpy(dtype=float)
            for _, cell_frame in block_frame.groupby("layer", sort=True, dropna=False)
        ]
        hierarchy.append((tuple(block), cells))

    original_block_means = [
        float(np.mean([float(values.mean()) for values in cells]))
        for _, cells in hierarchy
    ]
    rng = np.random.default_rng(seed)
    replicates = np.empty(draws, dtype=np.float64)
    n_blocks = len(hierarchy)
    for draw in range(draws):
        sampled_block_means: list[float] = []
        for block_index in rng.integers(0, n_blocks, size=n_blocks):
            cells = hierarchy[int(block_index)][1]
            n_cells = len(cells)
            sampled_cell_means: list[float] = []
            for cell_index in rng.integers(0, n_cells, size=n_cells):
                values = cells[int(cell_index)]
                sampled = values[rng.integers(0, len(values), size=len(values))]
                sampled_cell_means.append(float(sampled.mean()))
            sampled_block_means.append(float(np.mean(sampled_cell_means)))
        replicates[draw] = float(np.mean(sampled_block_means))

    tail = (1.0 - confidence) / 2.0
    ci_low, ci_high = np.quantile(replicates, (tail, 1.0 - tail))
    return {
        "method": "equal_block_hierarchical_percentile_bootstrap",
        "cluster_unit": "model_key_task",
        "hierarchy": ["model_task", "layer", "pair"],
        "estimand_weighting": "equal_model_task_blocks",
        "draws": int(draws),
        "seed": int(seed),
        "confidence": float(confidence),
        "point_estimate": float(np.mean(original_block_means)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_blocks": n_blocks,
        "n_cells": int(sum(len(cells) for _, cells in hierarchy)),
        "n_pairs": len(paired),
    }


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


def _floor_summary(
    alterrep: pd.DataFrame, floor: float, *, expected_blocks: int
) -> dict[str, Any]:
    paired, excluded = _floor_pair_data(alterrep, floor)
    coverage = _coverage_at_floor(alterrep, paired)
    requested_pairs = int(alterrep[list(_PAIR_COLUMNS)].drop_duplicates().shape[0])
    requested_blocks = int(alterrep[list(_BLOCK_COLUMNS)].drop_duplicates().shape[0])
    if paired.empty:
        raise AnalysisValidationError(
            f"floor {floor!r} expected {expected_blocks} model-task blocks, "
            "observed none"
        )
    _cells, blocks, means = _hierarchical_means(paired)
    sign_flip = exact_paired_sign_flip(
        blocks["gap"].to_numpy(), expected_blocks=expected_blocks
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
            "Floor-specific hierarchical intervals were not preregistered for "
            "this post-hoc curve. To avoid multiplying exploratory interval "
            "estimates, this sensitivity reports the available-case point "
            "estimate, coverage, and exact block sign-flip result; the raw-row "
            "hierarchical interval is reported separately."
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
    expected_alterrep_keys: Iterable[Any] | None = None,
    expected_pairs: int = _LOCKED_EXPECTED_PAIRS,
    expected_cells: int = _LOCKED_EXPECTED_CELLS,
    expected_blocks: int = _LOCKED_EXPECTED_BLOCKS,
    draws: int = 10_000,
    seed: int = 20260830,
) -> dict[str, Any]:
    """Return deterministic post-hoc raw-drop and floor-robustness analyses.

    ``expected_alterrep_keys`` is the caller's exact planned six-field row-key
    universe.  When it is omitted, the summary enforces the locked 300-pair,
    60-cell, 12-block counts but reports that only count completeness was
    checked.  Explicit count overrides support deliberately small fixtures.
    """

    draws = _validated_expected_count(draws, "draws")
    seed = _validated_seed(seed)
    provenance = _source_provenance(rows)
    alterrep, validation = _canonical_alterrep_rows(
        rows,
        expected_alterrep_keys=expected_alterrep_keys,
        expected_pairs=expected_pairs,
        expected_cells=expected_cells,
        expected_blocks=expected_blocks,
    )
    alterrep["raw_drop"] = raw_target_drop(alterrep)
    raw_pairs = _paired_values(alterrep, "raw_drop")
    raw_cells, raw_blocks, raw_means = _hierarchical_means(raw_pairs)
    bootstrap = _equal_block_hierarchical_bootstrap(
        raw_pairs,
        draws=draws,
        seed=seed,
    )
    bootstrap["equal_block_point_estimate"] = raw_means["gap"]
    raw_sign_flip = exact_paired_sign_flip(
        raw_blocks["gap"].to_numpy(), expected_blocks=expected_blocks
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
    floor_curve = [
        _floor_summary(alterrep, floor, expected_blocks=expected_blocks)
        for floor in FLOOR_GRID
    ]
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


_CONSTRUCT_CELL_COLUMNS = ("model_key", "model_id", "task", "layer")
_CONSTRUCT_REQUIRED_COLUMNS = {
    *_CONSTRUCT_CELL_COLUMNS,
    "row_kind",
    "evaluation_family",
    "decoder_seed",
    "label",
    "group_id",
    "n_examples",
    "correct_count",
    "split_manifest_sha256",
    "decoder_checkpoint_sha256",
    "edit_id",
    "edit_object",
    "architecture",
    "candidate_seed",
    "edit_hash",
    "status",
    "failure_stage",
    "failure_reason",
}
_CONSTRUCT_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CONSTRUCT_EDIT = re.compile(r"dcand_crossfit-(linear|mlp|mka)-seed(0|[1-9]|1[0-9])\Z")


@dataclass(frozen=True)
class _PreparedConstructCell:
    cell: ConstructCell
    groups: tuple[str, ...]
    group_n: np.ndarray
    baseline_target_correct: np.ndarray
    baseline_control_correct: np.ndarray
    post_target_correct: np.ndarray
    post_control_correct: np.ndarray
    split_manifest_sha256: str


def _require_nonblank_strings(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        invalid = frame[column].map(
            lambda value: not isinstance(value, str) or not value.strip()
        )
        if invalid.any():
            bad = frame.index[invalid].tolist()
            raise AnalysisValidationError(
                f"construct group rows require nonblank {column}; invalid rows {bad[:10]}"
            )


def _strict_integer_series(
    frame: pd.DataFrame,
    column: str,
    *,
    minimum: int,
    allow_integral_float: bool = False,
) -> pd.Series:
    converted: list[int] = []
    invalid: list[Any] = []
    for index, value in frame[column].items():
        if isinstance(value, (bool, np.bool_)):
            invalid.append(index)
            continue
        if isinstance(value, (int, np.integer)) or (
            allow_integral_float
            and isinstance(value, (float, np.floating))
            and np.isfinite(value)
            and float(value).is_integer()
        ):
            integer = int(value)
        else:
            invalid.append(index)
            continue
        if integer < minimum:
            invalid.append(index)
            continue
        converted.append(integer)
    if invalid:
        raise AnalysisValidationError(
            f"construct {column} must contain integers >= {minimum}; "
            f"invalid rows {invalid[:10]}"
        )
    return pd.Series(converted, index=frame.index, dtype=np.int64)


def _validate_sha_column(frame: pd.DataFrame, column: str) -> None:
    invalid = frame[column].map(
        lambda value: (
            not isinstance(value, str) or _CONSTRUCT_SHA256.fullmatch(value) is None
        )
    )
    if invalid.any():
        bad = frame.index[invalid].tolist()
        raise AnalysisValidationError(
            f"construct {column} must contain lowercase SHA-256 values; "
            f"invalid rows {bad[:10]}"
        )


def _validate_construct_group_rows(
    rows: pd.DataFrame | Path | str | Iterable[Mapping[str, Any]],
    *,
    config: ExtensionConfig,
) -> tuple[pd.DataFrame, dict[str, str]]:
    frame = load_rows(rows)
    _require_unique_column_labels(frame)
    missing = sorted(_CONSTRUCT_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise AnalysisValidationError(
            f"construct group rows are missing required columns: {missing}"
        )
    if frame.empty:
        raise AnalysisValidationError(
            "construct group rows require explicit coverage of all 12 locked cells"
        )
    _require_nonblank_strings(
        frame,
        (
            "model_key",
            "model_id",
            "task",
            "row_kind",
            "evaluation_family",
            "status",
        ),
    )
    _strict_integer_series(frame, "layer", minimum=1)
    decoder_seed = _strict_integer_series(frame, "decoder_seed", minimum=0)
    if (decoder_seed != 0).any():
        raise AnalysisValidationError(
            "construct confirmatory rows require fresh-linear decoder_seed 0"
        )
    if (frame["evaluation_family"] != "fresh_linear").any():
        raise AnalysisValidationError(
            "construct confirmatory rows require evaluation_family fresh_linear"
        )
    if not frame["row_kind"].isin(("baseline", "post_edit", "cell_status")).all():
        raise AnalysisValidationError(
            "construct row_kind must be baseline, post_edit, or cell_status"
        )
    if not frame["status"].isin(("ok", "failed", "nonestimable")).all():
        raise AnalysisValidationError("construct group rows contain an invalid status")

    cell_slugs = pd.Series(pd.NA, index=frame.index, dtype="string")
    for cell in config.all_cells:
        mask = (
            (frame["model_key"] == cell.model_key)
            & (frame["model_id"] == cell.model_id)
            & (frame["task"] == cell.task)
            & (frame["layer"] == cell.layer)
        )
        cell_slugs.loc[mask] = cell.slug
    unexpected = cell_slugs.isna()
    if unexpected.any():
        sample = frame.loc[unexpected, list(_CONSTRUCT_CELL_COLUMNS)].head(5)
        raise AnalysisValidationError(
            "construct group rows contain unexpected cells: "
            f"{sample.to_dict(orient='records')!r}"
        )
    frame["_cell_slug"] = pd.Categorical(
        cell_slugs,
        categories=[cell.slug for cell in config.all_cells],
    )
    observed_slugs = set(cell_slugs)
    missing_cells = [
        cell.slug for cell in config.all_cells if cell.slug not in observed_slugs
    ]
    if missing_cells:
        raise AnalysisValidationError(
            "construct group rows require exact cell coverage; missing explicit "
            f"cell records: {missing_cells}"
        )

    blocked: dict[str, str] = {}
    for slug, group in frame.groupby("_cell_slug", sort=False, observed=True):
        statuses = set(group["status"])
        if statuses == {"ok"}:
            if (group["row_kind"] == "cell_status").any():
                raise AnalysisValidationError(
                    f"construct cell {slug} cannot use cell_status with status ok"
                )
            continue
        if len(group) != 1 or len(statuses) != 1:
            raise AnalysisValidationError(
                f"construct cell {slug} requires exactly one explicit non-ok status row"
            )
        status_row = group.iloc[0]
        if status_row["row_kind"] != "cell_status":
            raise AnalysisValidationError(
                f"construct cell {slug} non-ok row must use row_kind cell_status"
            )
        for field in ("failure_stage", "failure_reason"):
            value = status_row[field]
            if not isinstance(value, str) or not value.strip():
                raise AnalysisValidationError(
                    f"construct cell {slug} non-ok row requires {field}"
                )
        for field in (
            "label",
            "group_id",
            "split_manifest_sha256",
            "decoder_checkpoint_sha256",
            "edit_id",
            "edit_object",
            "architecture",
            "candidate_seed",
            "edit_hash",
        ):
            if not pd.isna(status_row[field]):
                raise AnalysisValidationError(
                    f"construct cell {slug} non-ok row must leave {field} null"
                )
        for field in ("n_examples", "correct_count"):
            value = status_row[field]
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) != 0
            ):
                raise AnalysisValidationError(
                    f"construct cell {slug} non-ok row requires integer {field}=0"
                )
        blocked[str(slug)] = (
            f"{status_row['status']}: {status_row['failure_stage']}: "
            f"{status_row['failure_reason']}"
        )

    ok = frame.loc[frame["status"] == "ok"]
    if not ok.empty:
        if not ok["row_kind"].isin(("baseline", "post_edit")).all():
            raise AnalysisValidationError(
                "ok construct rows must be baseline or post_edit"
            )
        if not ok["label"].isin(("target", "control")).all():
            raise AnalysisValidationError(
                "ok construct label must be target or control"
            )
        _require_nonblank_strings(ok, ("label", "group_id"))
        if ok[["failure_stage", "failure_reason"]].notna().any().any():
            raise AnalysisValidationError(
                "ok construct rows must leave failure_stage and failure_reason null"
            )
        n_examples = _strict_integer_series(ok, "n_examples", minimum=1)
        correct_count = _strict_integer_series(ok, "correct_count", minimum=0)
        if (correct_count > n_examples).any():
            raise AnalysisValidationError(
                "construct correct_count cannot exceed n_examples"
            )
        _validate_sha_column(ok, "split_manifest_sha256")
        _validate_sha_column(ok, "decoder_checkpoint_sha256")

        baseline = ok["row_kind"] == "baseline"
        for column in (
            "edit_id",
            "edit_object",
            "architecture",
            "candidate_seed",
            "edit_hash",
        ):
            if ok.loc[baseline, column].notna().any():
                raise AnalysisValidationError(
                    f"construct baseline rows must leave {column} null"
                )
        post = ~baseline
        _require_nonblank_strings(
            ok.loc[post],
            ("edit_id", "edit_object", "architecture", "edit_hash"),
        )
        _validate_sha_column(ok.loc[post], "edit_hash")
        _strict_integer_series(
            ok.loc[post],
            "candidate_seed",
            minimum=0,
            allow_integral_float=True,
        )

        edit_key = ok["edit_id"].where(post, "__baseline__")
        duplicate_columns = [
            "_cell_slug",
            "row_kind",
            "evaluation_family",
            "decoder_seed",
            "label",
            "group_id",
        ]
        duplicate_probe = ok[duplicate_columns].copy()
        duplicate_probe["_edit_key"] = edit_key
        duplicates = duplicate_probe.duplicated(keep=False)
        if duplicates.any():
            raise AnalysisValidationError("duplicate construct group-row key")
    return frame, blocked


def _prepare_construct_cell(
    frame: pd.DataFrame,
    cell: ConstructCell,
) -> _PreparedConstructCell:
    cell_rows = frame.loc[
        (frame["_cell_slug"] == cell.slug) & (frame["status"] == "ok")
    ].copy()
    if cell_rows.empty:
        raise AnalysisValidationError(f"construct cell {cell.slug} has no ok rows")
    split_hashes = set(cell_rows["split_manifest_sha256"])
    if len(split_hashes) != 1:
        raise AnalysisValidationError(
            f"construct cell {cell.slug} has inconsistent split manifest hashes"
        )
    baseline = cell_rows.loc[cell_rows["row_kind"] == "baseline"].copy()
    post = cell_rows.loc[cell_rows["row_kind"] == "post_edit"].copy()
    if baseline.empty or post.empty:
        raise AnalysisValidationError(
            f"construct cell {cell.slug} requires baseline and post-edit rows"
        )
    for label in ("target", "control"):
        hashes = set(
            baseline.loc[baseline["label"] == label, "decoder_checkpoint_sha256"]
        )
        if len(hashes) != 1:
            raise AnalysisValidationError(
                f"construct cell {cell.slug} baseline {label} rows must share "
                "one decoder checkpoint hash"
            )
    observed_edits = set(post["edit_id"])
    if observed_edits != set(LOCKED_CONSTRUCT_CANDIDATE_EDIT_IDS):
        missing = sorted(set(LOCKED_CONSTRUCT_CANDIDATE_EDIT_IDS) - observed_edits)
        unexpected = sorted(observed_edits - set(LOCKED_CONSTRUCT_CANDIDATE_EDIT_IDS))
        raise AnalysisValidationError(
            f"construct cell {cell.slug} requires the exact 60 candidate edits; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    for edit_id, group in post.groupby("edit_id", sort=False):
        match = _CONSTRUCT_EDIT.fullmatch(str(edit_id))
        if match is None:
            raise AnalysisValidationError(
                f"invalid construct edit identity: {edit_id!r}"
            )
        expected_architecture = match.group(1)
        expected_seed = int(match.group(2))
        if set(group["edit_object"]) != {"dcand_crossfit"}:
            raise AnalysisValidationError(
                f"construct edit {edit_id} has an invalid edit object"
            )
        if set(group["architecture"]) != {expected_architecture}:
            raise AnalysisValidationError(
                f"construct edit {edit_id} has inconsistent architecture"
            )
        seeds = {int(value) for value in group["candidate_seed"]}
        if seeds != {expected_seed}:
            raise AnalysisValidationError(
                f"construct edit {edit_id} has inconsistent candidate seed"
            )
        if len(set(group["edit_hash"])) != 1:
            raise AnalysisValidationError(
                f"construct edit {edit_id} has inconsistent edit hashes"
            )
        for label in ("target", "control"):
            decoder_hashes = set(
                group.loc[group["label"] == label, "decoder_checkpoint_sha256"]
            )
            if len(decoder_hashes) != 1:
                raise AnalysisValidationError(
                    f"construct edit {edit_id}/{label} rows must share one "
                    "decoder checkpoint hash"
                )

    baseline_groups: dict[str, set[str]] = {}
    for label in ("target", "control"):
        label_rows = baseline.loc[baseline["label"] == label]
        baseline_groups[label] = set(label_rows["group_id"])
        if not baseline_groups[label]:
            raise AnalysisValidationError(
                f"construct cell {cell.slug} has no {label} baseline groups"
            )
    if baseline_groups["target"] != baseline_groups["control"]:
        raise AnalysisValidationError(
            f"construct cell {cell.slug} baseline target/control groups differ"
        )
    groups = tuple(sorted(baseline_groups["target"]))

    def ordered(label_rows: pd.DataFrame) -> pd.DataFrame:
        indexed = label_rows.set_index("group_id")
        if not indexed.index.is_unique:
            raise AnalysisValidationError(
                f"construct cell {cell.slug} has duplicate group rows"
            )
        if set(indexed.index) != set(groups):
            raise AnalysisValidationError(
                f"construct cell {cell.slug} has incomplete group coverage"
            )
        return indexed.loc[list(groups)]

    baseline_target = ordered(baseline.loc[baseline["label"] == "target"])
    baseline_control = ordered(baseline.loc[baseline["label"] == "control"])
    group_n = baseline_target["n_examples"].to_numpy(dtype=np.float64)
    if not np.array_equal(
        baseline_control["n_examples"].to_numpy(dtype=np.float64), group_n
    ):
        raise AnalysisValidationError(
            f"construct cell {cell.slug} baseline label group sizes differ"
        )

    post_target = np.empty(
        (len(LOCKED_CONSTRUCT_CANDIDATE_EDIT_IDS), len(groups)), dtype=np.float64
    )
    post_control = np.empty_like(post_target)
    for edit_index, edit_id in enumerate(LOCKED_CONSTRUCT_CANDIDATE_EDIT_IDS):
        edit_rows = post.loc[post["edit_id"] == edit_id]
        for label, destination in (
            ("target", post_target),
            ("control", post_control),
        ):
            selected = ordered(edit_rows.loc[edit_rows["label"] == label])
            if not np.array_equal(
                selected["n_examples"].to_numpy(dtype=np.float64), group_n
            ):
                raise AnalysisValidationError(
                    f"construct cell {cell.slug}/{edit_id}/{label} group sizes differ"
                )
            destination[edit_index] = selected["correct_count"].to_numpy(
                dtype=np.float64
            )
    return _PreparedConstructCell(
        cell=cell,
        groups=groups,
        group_n=group_n,
        baseline_target_correct=baseline_target["correct_count"].to_numpy(
            dtype=np.float64
        ),
        baseline_control_correct=baseline_control["correct_count"].to_numpy(
            dtype=np.float64
        ),
        post_target_correct=post_target,
        post_control_correct=post_control,
        split_manifest_sha256=next(iter(split_hashes)),
    )


def _construct_point_endpoints(
    cell: _PreparedConstructCell,
) -> tuple[dict[str, float] | None, dict[str, float]]:
    total = float(cell.group_n.sum())
    target_baseline = float(cell.baseline_target_correct.sum() / total)
    control_baseline = float(cell.baseline_control_correct.sum() / total)
    baselines = {
        "target_accuracy": target_baseline,
        "control_accuracy": control_baseline,
    }
    if target_baseline <= 0.5 or control_baseline <= 0.5:
        return None, baselines
    target_post = cell.post_target_correct.sum(axis=1) / total
    control_post = cell.post_control_correct.sum(axis=1) / total
    endpoints = {
        "median_target_post_edit_accuracy": float(np.median(target_post)),
        "median_target_recovery_ratio": float(
            np.median((target_post - 0.5) / (target_baseline - 0.5))
        ),
        "median_control_retention_ratio": float(
            np.median((control_post - 0.5) / (control_baseline - 0.5))
        ),
    }
    return endpoints, baselines


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return deterministic Holm step-down adjusted p-values."""

    if not p_values:
        raise AnalysisValidationError("Holm adjustment requires p-values")
    checked: list[tuple[str, float]] = []
    for key, value in p_values.items():
        if not isinstance(key, str) or not key:
            raise AnalysisValidationError("Holm p-value keys must be nonblank strings")
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise AnalysisValidationError(f"Holm p-value for {key} is not numeric")
        numeric = float(value)
        if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise AnalysisValidationError(f"Holm p-value for {key} is outside [0, 1]")
        checked.append((key, numeric))
    ordered = sorted(checked, key=lambda item: (item[1], item[0]))
    family_size = len(ordered)
    running = 0.0
    adjusted: dict[str, float] = {}
    for rank, (key, value) in enumerate(ordered):
        candidate = min(1.0, (family_size - rank) * value)
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def _task_order(config: ExtensionConfig) -> tuple[str, ...]:
    return tuple(dict.fromkeys(cell.task for cell in config.confirmatory_cells))


def _resample_hash_header(task: str, groups: Sequence[str]) -> bytes:
    payload = bytearray(task.encode("utf-8"))
    payload.extend(b"\0")
    for group in groups:
        encoded = group.encode("utf-8")
        payload.extend(len(encoded).to_bytes(8, "little"))
        payload.extend(encoded)
    return bytes(payload)


def _task_bootstrap_seed(seed: int, task: str) -> int:
    digest = hashlib.sha256(
        f"reviewer_revision.construct_panel.PCG64.v1\0{seed}\0{task}".encode()
    ).digest()
    return int.from_bytes(digest[:16], "little")


def _construct_bootstrap(
    prepared: Mapping[str, _PreparedConstructCell],
    *,
    config: ExtensionConfig,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, str | None],
    dict[str, int],
]:
    draws = config.bootstrap_draws
    distributions: dict[str, dict[str, np.ndarray]] = {}
    invalid_counts: dict[str, dict[str, int]] = {}
    task_hashes: dict[str, str | None] = {}
    task_seeds: dict[str, int] = {}
    for slug in prepared:
        distributions[slug] = {
            "accuracy": np.empty(draws, dtype=np.float64),
            "target_recovery_ratio": np.empty(draws, dtype=np.float64),
            "control_retention_ratio": np.empty(draws, dtype=np.float64),
        }
        invalid_counts[slug] = {
            "target_baseline_at_or_below_chance": 0,
            "control_baseline_at_or_below_chance": 0,
        }

    for task in _task_order(config):
        task_seed = _task_bootstrap_seed(config.bootstrap_seed, task)
        task_seeds[task] = task_seed
        generator = np.random.Generator(np.random.PCG64(task_seed))
        task_cells = [
            cell
            for cell in config.confirmatory_cells
            if cell.task == task and cell.slug in prepared
        ]
        if not task_cells:
            task_hashes[task] = None
            continue
        reference = prepared[task_cells[0].slug]
        for cell in task_cells[1:]:
            candidate = prepared[cell.slug]
            if candidate.groups != reference.groups or not np.array_equal(
                candidate.group_n, reference.group_n
            ):
                raise AnalysisValidationError(
                    f"construct task {task} models do not share the same shared group universe"
                )
        group_count = len(reference.groups)
        hasher = hashlib.sha256(_resample_hash_header(task, reference.groups))
        batch_size = 256
        for start in range(0, draws, batch_size):
            stop = min(draws, start + batch_size)
            size = stop - start
            sampled = generator.integers(
                0,
                group_count,
                size=(size, group_count),
                dtype=np.int32,
            )
            weights = np.zeros((size, group_count), dtype=np.int32)
            rows_index = np.repeat(np.arange(size, dtype=np.int32), group_count)
            np.add.at(weights, (rows_index, sampled.reshape(-1)), 1)
            hasher.update(weights.astype("<i4", copy=False).tobytes(order="C"))
            weights_float = weights.astype(np.float64)
            for cell in task_cells:
                values = prepared[cell.slug]
                denominator = weights_float @ values.group_n
                target_baseline = (
                    weights_float @ values.baseline_target_correct
                ) / denominator
                control_baseline = (
                    weights_float @ values.baseline_control_correct
                ) / denominator
                target_post = (
                    weights_float @ values.post_target_correct.T
                ) / denominator[:, None]
                control_post = (
                    weights_float @ values.post_control_correct.T
                ) / denominator[:, None]
                endpoint = distributions[cell.slug]
                endpoint["accuracy"][start:stop] = np.median(target_post, axis=1)
                target_valid = target_baseline > 0.5
                target_ratio = np.full(size, -np.inf, dtype=np.float64)
                if target_valid.any():
                    target_ratio[target_valid] = np.median(
                        (target_post[target_valid] - 0.5)
                        / (target_baseline[target_valid, None] - 0.5),
                        axis=1,
                    )
                endpoint["target_recovery_ratio"][start:stop] = target_ratio
                invalid_counts[cell.slug]["target_baseline_at_or_below_chance"] += int(
                    (~target_valid).sum()
                )
                control_valid = control_baseline > 0.5
                control_ratio = np.full(size, -np.inf, dtype=np.float64)
                if control_valid.any():
                    control_ratio[control_valid] = np.median(
                        (control_post[control_valid] - 0.5)
                        / (control_baseline[control_valid, None] - 0.5),
                        axis=1,
                    )
                endpoint["control_retention_ratio"][start:stop] = control_ratio
                invalid_counts[cell.slug]["control_baseline_at_or_below_chance"] += int(
                    (~control_valid).sum()
                )
        task_hashes[task] = hasher.hexdigest()

    thresholds = dict(config.recovery_thresholds)
    outputs: dict[str, dict[str, Any]] = {}
    for slug, endpoint in distributions.items():
        endpoint_p_values: dict[str, float] = {}
        lower_bounds: dict[str, float | None] = {}
        lower_bound_finite: dict[str, bool] = {}
        for name, values in endpoint.items():
            threshold = thresholds[name]
            endpoint_p_values[name] = float(
                (1 + np.count_nonzero(values <= threshold)) / (draws + 1)
            )
            with np.errstate(invalid="ignore"):
                lower = float(
                    np.quantile(
                        values,
                        config.alpha,
                        method=config.bootstrap_quantile_method,
                    )
                )
            finite = bool(np.isfinite(lower))
            lower_bounds[name] = lower if finite else None
            lower_bound_finite[name] = finite
        outputs[slug] = {
            "endpoint_p_values": endpoint_p_values,
            "internal_cell_p_value": max(endpoint_p_values.values()),
            "marginal_lower_bounds": lower_bounds,
            "marginal_lower_bound_finite": lower_bound_finite,
            "invalid_bootstrap_draws": invalid_counts[slug],
        }
    return outputs, task_hashes, task_seeds


def _point_threshold_decision(
    endpoints: Mapping[str, float],
    thresholds: Mapping[str, float],
) -> tuple[dict[str, bool], bool]:
    decisions = {
        "accuracy": endpoints["median_target_post_edit_accuracy"]
        >= thresholds["accuracy"],
        "target_recovery_ratio": endpoints["median_target_recovery_ratio"]
        >= thresholds["target_recovery_ratio"],
        "control_retention_ratio": endpoints["median_control_retention_ratio"]
        >= thresholds["control_retention_ratio"],
    }
    return decisions, all(decisions.values())


def classify_construct_panel(
    rows: pd.DataFrame | Path | str | Iterable[Mapping[str, Any]],
    *,
    config: ExtensionConfig,
) -> dict[str, Any]:
    """Classify the fixed 11-cell construct panel from group sufficient statistics."""

    if not isinstance(config, ExtensionConfig):
        raise TypeError("config must be an ExtensionConfig")
    config.validate()
    frame, blocked = _validate_construct_group_rows(rows, config=config)
    prepared: dict[str, _PreparedConstructCell] = {}
    preparation_failures: dict[str, str] = {}
    for cell in config.all_cells:
        if cell.slug in blocked:
            preparation_failures[cell.slug] = blocked[cell.slug]
            continue
        prepared[cell.slug] = _prepare_construct_cell(frame, cell)

    confirmatory_prepared = {
        cell.slug: prepared[cell.slug]
        for cell in config.confirmatory_cells
        if cell.slug in prepared
    }
    bootstrap, task_hashes, task_seeds = _construct_bootstrap(
        confirmatory_prepared,
        config=config,
    )
    thresholds = dict(config.recovery_thresholds)
    cells: dict[str, dict[str, Any]] = {}
    raw_cell_p_values: dict[str, float] = {}
    point_pass: dict[str, bool] = {}

    for cell in config.all_cells:
        confirmatory = cell != config.pilot
        base_record: dict[str, Any] = {
            "cell": {
                "model_key": cell.model_key,
                "model_id": cell.model_id,
                "task": cell.task,
                "layer": cell.layer,
            },
            "confirmatory": confirmatory,
            "thresholds": thresholds,
        }
        if cell.slug not in prepared:
            record = {
                **base_record,
                "status": "nonestimable",
                "nonestimable_reason": preparation_failures[cell.slug],
            }
            if confirmatory:
                record.update(
                    {
                        "passes_locked_point_thresholds": False,
                        "internal_cell_p_value": 1.0,
                    }
                )
                raw_cell_p_values[cell.slug] = 1.0
                point_pass[cell.slug] = False
            else:
                record["inference_mode"] = "descriptive_only"
            cells[cell.slug] = record
            continue
        prepared_cell = prepared[cell.slug]
        endpoints, baselines = _construct_point_endpoints(prepared_cell)
        record = {
            **base_record,
            "n_groups": len(prepared_cell.groups),
            "n_candidate_edits": len(LOCKED_CONSTRUCT_CANDIDATE_EDIT_IDS),
            "baseline_accuracies": baselines,
            "split_manifest_sha256": prepared_cell.split_manifest_sha256,
        }
        if endpoints is None:
            record.update(
                {
                    "status": "nonestimable",
                    "nonestimable_reason": (
                        "target or control fresh-linear baseline is at or below chance"
                    ),
                }
            )
            if confirmatory:
                record.update(
                    {
                        "passes_locked_point_thresholds": False,
                        "internal_cell_p_value": 1.0,
                    }
                )
                raw_cell_p_values[cell.slug] = 1.0
                point_pass[cell.slug] = False
            else:
                record["inference_mode"] = "descriptive_only"
            cells[cell.slug] = record
            continue
        endpoint_decisions, passes_points = _point_threshold_decision(
            endpoints, thresholds
        )
        record.update(
            {
                "status": "ok",
                "endpoints": endpoints,
                "point_threshold_decisions": endpoint_decisions,
                "passes_locked_point_thresholds": passes_points,
            }
        )
        if not confirmatory:
            record["inference_mode"] = "descriptive_only"
            cells[cell.slug] = record
            continue
        inference = bootstrap[cell.slug]
        record.update(
            {
                **inference,
                "within_cell_combination": "intersection_union_max_endpoint_p",
                "task_resample_sha256": task_hashes[cell.task],
                "lower_bound_scope": "marginal",
                "lower_bound_multiplicity_adjusted": False,
                "lower_bound_simultaneous": False,
            }
        )
        raw_cell_p_values[cell.slug] = inference["internal_cell_p_value"]
        point_pass[cell.slug] = passes_points
        cells[cell.slug] = record

    if len(raw_cell_p_values) != config.family_size:
        raise AnalysisValidationError(
            "construct panel did not preserve the fixed 11-cell inference family"
        )
    adjusted = holm_adjust(raw_cell_p_values)
    for cell in config.confirmatory_cells:
        record = cells[cell.slug]
        adjusted_p = adjusted[cell.slug]
        passes_inference = adjusted_p <= config.alpha
        record["holm_adjusted_cell_p_value"] = adjusted_p
        record["passes_holm_adjusted_inference"] = passes_inference
        record["passes_locked_confirmatory_rule"] = (
            point_pass[cell.slug] and passes_inference
        )

    return {
        "schema": "reviewer_revision.construct_panel_inference.v1",
        "schema_version": 1,
        "status": "ok",
        "pilot": dict(cells[config.pilot.slug]),
        "confirmatory_cell_count": len(config.confirmatory_cells),
        "nonestimable_confirmatory_cell_count": sum(
            cells[cell.slug]["status"] == "nonestimable"
            for cell in config.confirmatory_cells
        ),
        "passing_confirmatory_cell_count": sum(
            bool(cells[cell.slug]["passes_locked_confirmatory_rule"])
            for cell in config.confirmatory_cells
        ),
        "cells": cells,
        "bootstrap": {
            "draws": config.bootstrap_draws,
            "seed": config.bootstrap_seed,
            "bit_generator": config.bootstrap_bit_generator,
            "resampling_unit": "final_test_group_id",
            "shared_resamples_across_models_within_task": True,
            "resample_algorithm": "PCG64 integers followed by exact multiplicity counts",
            "task_seed_derivation": (
                "first 128 little-endian bits of SHA256("
                "reviewer_revision.construct_panel.PCG64.v1\\0seed\\0task)"
            ),
            "task_seeds": task_seeds,
            "task_resample_sha256": task_hashes,
        },
        "multiplicity": {
            "method": "holm_one_sided",
            "family_size": config.family_size,
            "alpha": config.alpha,
        },
        "lower_bounds": {
            "confidence_level": config.lower_bound_confidence_level,
            "quantile": config.alpha,
            "quantile_method": config.bootstrap_quantile_method,
            "scope": config.lower_bound_scope,
            "multiplicity_adjusted": config.lower_bound_multiplicity_adjusted,
            "simultaneous": config.lower_bound_simultaneous,
        },
        "endpoint_tail_p_value": config.endpoint_tail_p_value,
        "within_cell_combination": config.within_cell_combination,
    }


__all__ = [
    "FLOOR_GRID",
    "LOCKED_CONSTRUCT_CANDIDATE_EDIT_IDS",
    "classify_construct_panel",
    "holm_adjust",
    "raw_target_drop",
    "summarize_floor_robustness",
    "target_damage_at_floor",
]
