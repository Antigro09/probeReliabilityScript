from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.reviewer_revision.analysis import AnalysisValidationError
from src.reviewer_revision.extension_analysis import (
    FLOOR_GRID,
    raw_target_drop,
    summarize_floor_robustness,
    target_damage_at_floor,
)

_PLAN_COUNTS = {"expected_pairs": 2, "expected_cells": 2, "expected_blocks": 2}
_HIERARCHY_COUNTS = {
    "expected_pairs": 3,
    "expected_cells": 3,
    "expected_blocks": 2,
}
_ONE_COUNTS = {"expected_pairs": 1, "expected_cells": 1, "expected_blocks": 1}


def _pair_rows(
    *,
    model: str,
    task: str,
    layer: int,
    pair_seed: int,
    matched: tuple[float, float],
    split: tuple[float, float],
    edit_hash: str | None = None,
    status: str = "ok",
) -> list[dict[str, object]]:
    shared_hash = edit_hash or f"edit-{model}-{task}-{layer}-{pair_seed}"
    rows: list[dict[str, object]] = []
    for condition, (pre, post) in (("matched", matched), ("split", split)):
        rows.append(
            {
                "model_key": model,
                "task": task,
                "layer": layer,
                "pair_seed": pair_seed,
                "method": "alterrep",
                "condition": condition,
                "target_acc_pre": pre,
                "target_acc_post": post,
                "chance": 0.5,
                "status": status,
                "edit_hash": shared_hash,
            }
        )
    return rows


def _plan_fixture_rows() -> list[dict[str, object]]:
    rows = _pair_rows(
        model="bert",
        task="sva",
        layer=6,
        pair_seed=0,
        matched=(0.90, 0.40),
        split=(0.80, 0.60),
    )
    rows.extend(
        _pair_rows(
            model="gpt2",
            task="sst2",
            layer=6,
            pair_seed=0,
            matched=(0.64, 0.44),
            split=(0.64, 0.54),
        )
    )
    return rows


def _real_hierarchy_fixture_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model, layer, gap in (
        ("model-a", 1, 0.2),
        ("model-a", 2, 0.4),
        ("model-b", 1, 0.8),
    ):
        split_drop = 0.1
        matched_drop = split_drop + gap
        rows.extend(
            _pair_rows(
                model=model,
                task="shared-task",
                layer=layer,
                pair_seed=0,
                matched=(0.9, 0.9 - matched_drop),
                split=(0.9, 0.9 - split_drop),
            )
        )
    return rows


def _alterrep_keys(
    rows: list[dict[str, object]],
) -> list[tuple[object, ...]]:
    columns = (
        "model_key",
        "task",
        "layer",
        "pair_seed",
        "method",
        "condition",
    )
    return [tuple(row[column] for column in columns) for row in rows]


def test_public_row_helpers_use_raw_drop_and_exact_floor_boundaries() -> None:
    floor = 0.55
    below = np.nextafter(floor, -np.inf)
    above = np.nextafter(floor, np.inf)
    frame = pd.DataFrame(
        {
            "target_acc_pre": [below, floor, above],
            "target_acc_post": [0.5, 0.5, 0.5],
            "chance": [0.5, 0.5, 0.5],
        }
    )

    raw = raw_target_drop(frame)
    damage = target_damage_at_floor(frame, floor)

    assert raw.tolist() == pytest.approx([below - 0.5, 0.05, above - 0.5])
    assert np.isnan(damage.iloc[0])
    assert damage.iloc[1] == pytest.approx(1.0)
    assert damage.iloc[2] == pytest.approx(1.0)


def test_target_damage_helper_requires_exact_chance() -> None:
    frame = pd.DataFrame(
        {
            "target_acc_pre": [0.8],
            "target_acc_post": [0.7],
            "chance": [np.nextafter(0.5, np.inf)],
        }
    )

    with pytest.raises(AnalysisValidationError, match="chance.*exactly 0.5"):
        target_damage_at_floor(frame, 0.55)


def test_floor_robustness_includes_every_raw_drop_pair() -> None:
    summary = summarize_floor_robustness(
        pd.DataFrame(_plan_fixture_rows()), draws=200, seed=7, **_PLAN_COUNTS
    )

    raw = summary["full_case_raw_drop"]
    assert summary["raw_drop"] == raw
    assert raw["pairs"] == 2
    assert raw["cells"] == 2
    assert raw["model_task_blocks"] == 2
    assert raw["matched_mean"] == pytest.approx(0.35)
    assert raw["split_mean"] == pytest.approx(0.15)
    assert raw["gap"] == pytest.approx(0.20)
    assert raw["post_hoc_sensitivity"] is True
    assert raw["exact_sign_flip"]["n_blocks"] == 2
    assert summary["validation"]["completeness_mode"] == "expected_counts_only"
    assert summary["validation"]["external_key_universe_checked"] is False
    assert summary["validation"]["keys_complete"] is None
    assert summary["validation"]["expected_count_source"] == "explicit_overrides"


def test_raw_drop_uses_equal_model_task_blocks_not_equal_cells() -> None:
    summary = summarize_floor_robustness(
        _real_hierarchy_fixture_rows(),
        draws=50,
        seed=17,
        **_HIERARCHY_COUNTS,
    )

    raw = summary["raw_drop"]
    assert raw["matched_mean"] == pytest.approx(0.65)
    assert raw["split_mean"] == pytest.approx(0.10)
    assert raw["gap"] == pytest.approx(0.55)
    assert raw["gap"] != pytest.approx((0.2 + 0.4 + 0.8) / 3.0)


def test_locked_default_counts_reject_a_small_observed_derived_universe() -> None:
    with pytest.raises(AnalysisValidationError, match="expected 300 pairs"):
        summarize_floor_robustness(_plan_fixture_rows(), draws=10, seed=1)


def test_external_key_universe_is_reported_as_the_completeness_basis() -> None:
    rows = _plan_fixture_rows()
    summary = summarize_floor_robustness(
        rows,
        expected_alterrep_keys=_alterrep_keys(rows),
        draws=20,
        seed=5,
        **_PLAN_COUNTS,
    )

    validation = summary["validation"]
    assert validation["completeness_mode"] == "external_key_universe"
    assert validation["external_key_universe_checked"] is True
    assert validation["keys_complete"] is True
    assert validation["expected_key_count"] == 4
    assert validation["observed_derived_key_universe"] is False


def test_external_key_universe_rejects_a_deleted_whole_pair() -> None:
    planned = _plan_fixture_rows()
    observed = [row for row in planned if row["model_key"] != "gpt2"]

    with pytest.raises(AnalysisValidationError, match="missing expected keys"):
        summarize_floor_robustness(
            observed,
            expected_alterrep_keys=_alterrep_keys(planned),
            draws=10,
            seed=1,
            **_PLAN_COUNTS,
        )


def test_expected_counts_reject_a_deleted_whole_cell() -> None:
    rows = _real_hierarchy_fixture_rows()
    observed = [
        row for row in rows if not (row["model_key"] == "model-a" and row["layer"] == 2)
    ]

    with pytest.raises(AnalysisValidationError, match="expected 3 cells"):
        summarize_floor_robustness(
            observed,
            draws=10,
            seed=1,
            expected_pairs=2,
            expected_cells=3,
            expected_blocks=2,
        )


def test_expected_counts_reject_a_deleted_whole_block() -> None:
    rows = _real_hierarchy_fixture_rows()
    observed = [row for row in rows if row["model_key"] != "model-b"]

    with pytest.raises(AnalysisValidationError, match="expected 2 model-task blocks"):
        summarize_floor_robustness(
            observed,
            draws=10,
            seed=1,
            expected_pairs=2,
            expected_cells=2,
            expected_blocks=2,
        )


def test_external_key_universe_rejects_same_count_replacement_key() -> None:
    planned = _plan_fixture_rows()
    observed = [dict(row) for row in planned]
    for row in observed:
        if row["model_key"] == "gpt2":
            row["model_key"] = "replacement-model"

    with pytest.raises(
        AnalysisValidationError, match="missing expected keys.*unexpected observed keys"
    ):
        summarize_floor_robustness(
            observed,
            expected_alterrep_keys=_alterrep_keys(planned),
            draws=10,
            seed=1,
            **_PLAN_COUNTS,
        )


def test_floor_sign_flip_uses_declared_expected_block_count() -> None:
    rows = _plan_fixture_rows()
    for row in rows:
        if row["model_key"] == "gpt2":
            row["target_acc_pre"] = 0.54
            row["target_acc_post"] = 0.50
            row["status"] = "pre_target_below_floor"

    with pytest.raises(
        AnalysisValidationError, match="sign-flip test expected 2 blocks, observed 1"
    ):
        summarize_floor_robustness(rows, draws=10, seed=1, **_PLAN_COUNTS)


def test_partial_identification_contains_available_case_gap() -> None:
    bounds = summarize_floor_robustness(
        _plan_fixture_rows(), draws=40, seed=7, **_PLAN_COUNTS
    )["partial_identification"]

    assert bounds["missing_pair_gap_domain"] == [-1.0, 1.0]
    assert bounds["lower"] <= bounds["available_case_gap"] <= bounds["upper"]
    assert bounds["is_confidence_interval"] is False
    assert bounds["label"] == "deterministic_partial_identification_bound"


def test_partial_identification_keeps_wholly_missing_cells_in_bounds() -> None:
    rows = _pair_rows(
        model="model-a",
        task="task-a",
        layer=1,
        pair_seed=0,
        matched=(0.8, 0.575),
        split=(0.8, 0.725),
    )
    rows.extend(
        _pair_rows(
            model="model-a",
            task="task-a",
            layer=2,
            pair_seed=0,
            matched=(0.54, 0.50),
            split=(0.54, 0.52),
            status="pre_target_below_floor",
        )
    )

    bounds = summarize_floor_robustness(
        rows,
        draws=30,
        seed=3,
        expected_pairs=2,
        expected_cells=2,
        expected_blocks=1,
    )["partial_identification"]

    assert bounds["observed_pairs"] == 1
    assert bounds["missing_pairs"] == 1
    assert bounds["available_case_gap"] == pytest.approx(0.5)
    assert bounds["lower"] == pytest.approx(-0.25)
    assert bounds["upper"] == pytest.approx(0.75)
    assert bounds["planned_cells"] == 2
    assert bounds["wholly_missing_cells"] == 1


def test_floor_curve_excludes_a_pair_symmetrically() -> None:
    rows = _pair_rows(
        model="model-a",
        task="task-a",
        layer=1,
        pair_seed=0,
        matched=(np.nextafter(0.55, -np.inf), 0.50),
        split=(0.80, 0.70),
        status="pre_target_below_floor",
    )
    rows.extend(
        _pair_rows(
            model="model-a",
            task="task-a",
            layer=1,
            pair_seed=1,
            matched=(0.80, 0.65),
            split=(0.80, 0.725),
        )
    )

    summary = summarize_floor_robustness(
        rows,
        draws=30,
        seed=2,
        expected_pairs=2,
        expected_cells=1,
        expected_blocks=1,
    )
    at_locked_floor = next(
        row for row in summary["floor_curve"] if row["floor"] == 0.55
    )

    assert at_locked_floor["requested_pairs"] == 2
    assert at_locked_floor["analyzed_pairs"] == 1
    assert at_locked_floor["full_cells"] == 0
    assert at_locked_floor["partial_cells"] == 1
    assert at_locked_floor["missing_cells"] == 0
    assert at_locked_floor["excluded_pair_keys"] == [["model-a", "task-a", 1, 0]]
    assert at_locked_floor["excluded_pairs"][0]["reasons"] == [
        "matched_target_acc_pre_below_floor"
    ]
    assert at_locked_floor["matched_mean"] == pytest.approx(0.5)
    assert at_locked_floor["split_mean"] == pytest.approx(0.25)


def test_score_null_from_control_denominator_is_retained() -> None:
    rows = _pair_rows(
        model="model-a",
        task="task-a",
        layer=1,
        pair_seed=0,
        matched=(0.80, 0.65),
        split=(0.80, 0.725),
    )
    rows[0]["status"] = "pre_control_below_floor"

    summary = summarize_floor_robustness(rows, draws=20, seed=9, **_ONE_COUNTS)
    locked = next(row for row in summary["floor_curve"] if row["floor"] == 0.55)

    assert summary["validation"]["score_null_rows"] == 1
    assert summary["raw_drop"]["pairs"] == 1
    assert locked["analyzed_pairs"] == 1
    assert locked["gap"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda rows: rows.append(dict(rows[0])), "duplicate"),
        (lambda rows: rows.pop(1), "exactly matched and split"),
        (
            lambda rows: rows[1].__setitem__("edit_hash", "different-edit"),
            "edit_hash mismatch",
        ),
    ],
)
def test_pair_and_edit_hash_validation(
    mutate: Callable[[list[dict[str, object]]], object], message: str
) -> None:
    rows = _pair_rows(
        model="model-a",
        task="task-a",
        layer=1,
        pair_seed=0,
        matched=(0.8, 0.6),
        split=(0.8, 0.7),
    )
    mutate(rows)

    with pytest.raises(AnalysisValidationError, match=message):
        summarize_floor_robustness(rows, draws=10, seed=1, **_ONE_COUNTS)


@pytest.mark.parametrize("status", ["failed", "invalid"])
def test_hard_failures_are_rejected(status: str) -> None:
    rows = _plan_fixture_rows()
    rows[0]["status"] = status

    with pytest.raises(AnalysisValidationError, match="hard-failure"):
        summarize_floor_robustness(rows, draws=10, seed=1, **_PLAN_COUNTS)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda rows: [row.__setitem__("method", "project") for row in rows],
            "alterrep",
        ),
        (lambda rows: rows[0].__setitem__("condition", "other"), "condition"),
        (lambda rows: rows[0].__setitem__("model_key", np.nan), "model_key"),
        (lambda rows: rows[0].__setitem__("model_key", np.inf), "model_key"),
        (lambda rows: rows[0].__setitem__("layer", np.nan), "layer"),
        (
            lambda rows: rows[0].__setitem__("target_acc_pre", np.nan),
            "target_acc_pre",
        ),
        (
            lambda rows: rows[0].__setitem__("target_acc_post", np.inf),
            "target_acc_post",
        ),
        (
            lambda rows: rows[0].__setitem__("target_acc_pre", -0.01),
            "target_acc_pre",
        ),
        (
            lambda rows: rows[0].__setitem__("target_acc_post", 1.01),
            "target_acc_post",
        ),
        (lambda rows: rows[0].__setitem__("chance", 0.6), "chance"),
        (
            lambda rows: rows[0].__setitem__("chance", np.nextafter(0.5, np.inf)),
            "chance",
        ),
    ],
)
def test_invalid_schema_values_are_rejected(
    mutate: Callable[[list[dict[str, object]]], object], message: str
) -> None:
    rows = _plan_fixture_rows()
    mutate(rows)

    with pytest.raises(AnalysisValidationError, match=message):
        summarize_floor_robustness(rows, draws=10, seed=1, **_PLAN_COUNTS)


@pytest.mark.parametrize(
    "field, value",
    [
        ("model_key", 123),
        ("task", 456),
        ("method", 789),
        ("condition", 101),
        ("status", 112),
        ("edit_hash", 123),
        ("edit_hash", np.inf),
    ],
)
def test_text_schema_fields_require_actual_nonblank_strings(
    field: str, value: object
) -> None:
    rows = _plan_fixture_rows()
    for row in rows:
        row[field] = value

    with pytest.raises(AnalysisValidationError, match=rf"{field!r}.*nonblank strings"):
        summarize_floor_robustness(rows, draws=10, seed=1, **_PLAN_COUNTS)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("layer", True, "positive non-boolean integer"),
        ("layer", 1.0, "positive non-boolean integer"),
        ("layer", 1.5, "positive non-boolean integer"),
        ("layer", 0, "positive non-boolean integer"),
        ("pair_seed", False, "nonnegative non-boolean integer"),
        ("pair_seed", 0.0, "nonnegative non-boolean integer"),
        ("pair_seed", 0.5, "nonnegative non-boolean integer"),
        ("pair_seed", -1, "nonnegative non-boolean integer"),
    ],
)
def test_integral_keys_reject_bools_floats_and_out_of_domain_values(
    field: str, value: object, message: str
) -> None:
    rows = _plan_fixture_rows()
    for row in rows:
        row[field] = value

    with pytest.raises(AnalysisValidationError, match=message):
        summarize_floor_robustness(rows, draws=10, seed=1, **_PLAN_COUNTS)


def test_numpy_large_integer_pair_seed_is_preserved_exactly() -> None:
    large_seed = np.int64(2**53 + 1)
    rows = _pair_rows(
        model="model-a",
        task="task-a",
        layer=np.int64(1),
        pair_seed=large_seed,
        matched=(0.54, 0.50),
        split=(0.54, 0.52),
        status="pre_target_below_floor",
    )
    rows.extend(
        _pair_rows(
            model="model-a",
            task="task-a",
            layer=np.int64(1),
            pair_seed=np.int64(0),
            matched=(0.8, 0.65),
            split=(0.8, 0.725),
        )
    )

    summary = summarize_floor_robustness(
        rows,
        draws=10,
        seed=1,
        expected_pairs=2,
        expected_cells=1,
        expected_blocks=1,
    )
    locked = next(row for row in summary["floor_curve"] if row["floor"] == 0.55)

    assert locked["excluded_pair_keys"] == [["model-a", "task-a", 1, 2**53 + 1]]


def test_model_task_and_measurement_aliases_are_accepted() -> None:
    rows = pd.DataFrame(_plan_fixture_rows()).rename(
        columns={
            "model_key": "model",
            "task": "task_key",
            "condition": "scoring_condition",
            "target_acc_pre": "target_accuracy_pre",
            "target_acc_post": "target_accuracy_post",
        }
    )

    summary = summarize_floor_robustness(rows, draws=20, seed=5, **_PLAN_COUNTS)

    assert summary["raw_drop"]["pairs"] == 2
    assert summary["validation"]["column_aliases"]["model_key"] == "model"
    assert summary["validation"]["column_aliases"]["task"] == "task_key"


def test_summary_is_shuffle_rng_deterministic_and_json_serializable() -> None:
    rows = pd.DataFrame(_real_hierarchy_fixture_rows())
    shuffled = rows.sample(frac=1.0, random_state=123).reset_index(drop=True)

    first = summarize_floor_robustness(rows, draws=80, seed=19, **_HIERARCHY_COUNTS)
    second = summarize_floor_robustness(
        shuffled, draws=80, seed=19, **_HIERARCHY_COUNTS
    )

    assert first == second
    assert first["schema"] == "reviewer_revision.floor_robustness.v1"
    assert first["status"] == "ok"
    assert first["label"] == "post_hoc_sensitivity"
    assert first["replaces_registered_primary"] is False
    assert tuple(row["floor"] for row in first["floor_curve"]) == FLOOR_GRID
    json.dumps(first, sort_keys=True, allow_nan=False)


def test_path_input_records_content_hash(tmp_path: Path) -> None:
    path = tmp_path / "matched_split_rows.json"
    payload = json.dumps(_plan_fixture_rows(), sort_keys=True)
    path.write_text(payload, encoding="utf-8")

    summary = summarize_floor_robustness(path, draws=20, seed=4, **_PLAN_COUNTS)

    assert summary["provenance"]["input_kind"] == "path"
    assert (
        summary["provenance"]["source_sha256"]
        == hashlib.sha256(payload.encode("utf-8")).hexdigest()
    )
    assert summary["provenance"]["source_name"] == path.name
    assert "source_path" not in summary["provenance"]
    assert str(tmp_path.resolve()) not in json.dumps(summary, sort_keys=True)
