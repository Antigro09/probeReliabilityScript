from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.reviewer_revision.analysis import (
    AnalysisValidationError,
    exact_paired_sign_flip,
    hierarchical_bootstrap,
    load_rows,
    materialize_rows,
    summarize_construct_check,
    summarize_epsilon_sweep,
    summarize_matched_split,
    validate_expected_keys,
)


def test_key_completeness_counts_explicit_failure_as_covered() -> None:
    rows = pd.DataFrame(
        [
            {"model": "a", "seed": 0, "status": "ok"},
            {
                "model": "b",
                "seed": 0,
                "status": "failed",
                "failure_stage": "probe_fit",
                "failure_reason": "non-finite checkpoint",
            },
        ]
    )

    report = validate_expected_keys(
        rows,
        [("a", 0), ("b", 0)],
        ("model", "seed"),
    )

    assert report.is_complete
    assert report.expected_count == 2
    assert report.observed_count == 2
    assert report.ok_count == 1
    assert report.failure_count == 1
    assert report.missing_keys == ()


@pytest.mark.parametrize(
    "rows, message",
    [
        (
            [{"model": "a", "seed": 0, "status": "ok"}],
            "missing expected keys",
        ),
        (
            [
                {"model": "a", "seed": 0, "status": "ok"},
                {"model": "a", "seed": 0, "status": "ok"},
                {
                    "model": "b",
                    "seed": 0,
                    "status": "failed",
                    "failure_stage": "fit",
                    "failure_reason": "bad",
                },
            ],
            "duplicate observed keys",
        ),
        (
            [
                {"model": "a", "seed": 0, "status": "ok"},
                {
                    "model": "b",
                    "seed": 0,
                    "status": "failed",
                    "failure_stage": "fit",
                    "failure_reason": "",
                },
            ],
            "explicit failure_reason",
        ),
    ],
)
def test_key_completeness_refuses_missing_duplicate_or_unexplained_rows(
    rows: list[dict[str, object]], message: str
) -> None:
    with pytest.raises(AnalysisValidationError, match=message):
        validate_expected_keys(
            pd.DataFrame(rows),
            [("a", 0), ("b", 0)],
            ("model", "seed"),
        )


def test_materialize_and_load_rows_have_stable_key_order(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        [
            {"model": "b", "seed": 1, "value": 2.0},
            {"model": "a", "seed": 0, "value": 1.0},
        ]
    )
    csv_path = tmp_path / "rows.csv"
    parquet_path = tmp_path / "rows.parquet"

    materialized = materialize_rows(
        rows,
        csv_path=csv_path,
        parquet_path=parquet_path,
        key_columns=("model", "seed"),
    )

    assert list(materialized["model"]) == ["a", "b"]
    pd.testing.assert_frame_equal(load_rows(csv_path), materialized)
    pd.testing.assert_frame_equal(load_rows(parquet_path), materialized)


def test_materialize_rows_rejects_mismatched_paired_edit_hashes(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        [
            {
                "unit": "a",
                "condition": "matched",
                "edit_hash": "hash-one",
                "status": "ok",
            },
            {
                "unit": "a",
                "condition": "split",
                "edit_hash": "hash-two",
                "status": "ok",
            },
            {
                "unit": "reference-only",
                "condition": "reference",
                "edit_hash": "reference-hash",
                "status": "ok",
            },
        ]
    )

    with pytest.raises(AnalysisValidationError, match="edit_hash mismatch"):
        materialize_rows(
            rows,
            csv_path=tmp_path / "rows.csv",
            parquet_path=tmp_path / "rows.parquet",
            key_columns=("unit", "condition"),
        )


def test_materialize_rows_preserves_paired_score_null_with_same_edit(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        [
            {
                "unit": "a",
                "condition": "matched",
                "edit_hash": "same-edit",
                "status": "ok",
            },
            {
                "unit": "a",
                "condition": "split",
                "edit_hash": "same-edit",
                "status": "pre_target_below_floor",
                "failure_stage": "scoring",
                "failure_reason": "pre-edit target accuracy is below floor",
            },
        ]
    )

    materialized = materialize_rows(
        rows,
        csv_path=tmp_path / "rows.csv",
        parquet_path=tmp_path / "rows.parquet",
        key_columns=("unit", "condition"),
    )

    assert len(materialized) == 2
    assert set(materialized["status"]) == {"ok", "pre_target_below_floor"}


def _hierarchical_rows() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for block in range(12):
        for layer in (1, 2):
            for pair_seed in range(5):
                records.append(
                    {
                        "model": f"m{block // 2}",
                        "task": f"t{block % 2}",
                        "layer": layer,
                        "pair_seed": pair_seed,
                        "gap": 0.01 * block + 0.001 * layer + 0.0001 * pair_seed,
                    }
                )
    return pd.DataFrame(records)


def test_hierarchical_bootstrap_is_deterministic_at_locked_draw_count() -> None:
    rows = _hierarchical_rows()

    first = hierarchical_bootstrap(rows, draws=10_000, seed=20260830)
    second = hierarchical_bootstrap(rows, draws=10_000, seed=20260830)

    assert first == second
    assert first["draws"] == 10_000
    assert first["seed"] == 20260830
    assert first["cluster_unit"] == "model_task"
    assert first["hierarchy"] == ["model_task", "layer", "pair"]
    assert first["point_estimate"] == pytest.approx(rows["gap"].mean())
    assert first["ci_low"] <= first["ci_high"]


def test_exact_sign_flip_excludes_zeros_and_matches_archived_probability() -> None:
    result = exact_paired_sign_flip(
        [1.0] * 7 + [0.0] * 5,
        expected_blocks=12,
    )

    assert result["n_blocks"] == 12
    assert result["nonzero_count"] == 7
    assert result["zero_count"] == 5
    assert result["permutations"] == 128
    assert result["extreme_count"] == 2
    assert result["p_value"] == pytest.approx(0.015625)


def _matched_split_rows() -> tuple[pd.DataFrame, list[tuple[object, ...]]]:
    records: list[dict[str, object]] = []
    expected: list[tuple[object, ...]] = []
    for block in range(12):
        model = f"m{block // 2}"
        task = f"t{block % 2}"
        for depth_position, layer in enumerate((1, 2), start=1):
            for pair_seed in range(5):
                split = 0.40 + 0.01 * block
                matched = split + 0.20
                for condition, target_damage in (
                    ("matched", matched),
                    ("split", split),
                ):
                    target_accuracy_pre = 0.80 if condition == "matched" else 0.75
                    target_accuracy_post = 0.60 if condition == "matched" else 0.65
                    records.append(
                        {
                            "model": model,
                            "task": task,
                            "layer": layer,
                            "depth_position": depth_position,
                            "pair_seed": pair_seed,
                            "method": "alterrep",
                            "condition": condition,
                            "target_damage_C": target_damage,
                            "target_accuracy_pre": target_accuracy_pre,
                            "target_accuracy_post": target_accuracy_post,
                            "edit_hash": (
                                f"{model}:{task}:{layer}:{pair_seed}:alterrep"
                            ),
                            "status": "ok",
                        }
                    )
                    expected.append(
                        (model, task, layer, pair_seed, "alterrep", condition)
                    )
    return pd.DataFrame(records), expected


def test_matched_split_summary_uses_equal_cells_then_twelve_blocks() -> None:
    rows, expected = _matched_split_rows()

    summary = summarize_matched_split(
        rows,
        expected_keys=expected,
        key_columns=(
            "model",
            "task",
            "layer",
            "pair_seed",
            "method",
            "condition",
        ),
        expected_blocks=12,
        bootstrap_draws=400,
        bootstrap_seed=20260830,
        permutation_seed=20260831,
    )

    assert summary["included_units"] == {
        "rows": 240,
        "pairs": 120,
        "cells": 24,
        "model_task_blocks": 12,
    }
    assert summary["failed_units"] == 0
    assert summary["grand_mean_gap"] == pytest.approx(0.20)
    assert summary["grand_mean_matched"] - summary["grand_mean_split"] == pytest.approx(
        summary["grand_mean_gap"]
    )
    assert summary["fraction_cell_gaps_positive"] == 1.0
    assert summary["fraction_block_gaps_positive"] == 1.0
    assert summary["primary_exact_sign_flip"]["n_blocks"] == 12
    assert summary["primary_exact_sign_flip"]["p_value"] == pytest.approx(
        2 / 4096
    )
    assert summary["secondary_exact_sign_flip"]["n_cells"] == 24
    assert summary["secondary_exact_sign_flip"]["sensitivity_only"] is True
    assert summary["secondary_exact_sign_flip"]["p_value"] == pytest.approx(
        2 / (2**24)
    )
    assert set(summary["depth_position_medians"]) == {"1", "2"}
    assert summary["depth_position_medians"]["1"] == pytest.approx(0.2)
    assert summary["depth_position_medians"]["2"] == pytest.approx(0.2)
    assert set(summary["leave_one_model_out"]) == {
        "m0",
        "m1",
        "m2",
        "m3",
        "m4",
        "m5",
    }
    for interval in summary["leave_one_model_out"].values():
        assert interval["ci_low"] <= interval["mean"] <= interval["ci_high"]
        assert interval["method"] == "hierarchical_percentile_bootstrap"
    distributions = summary["target_accuracy_distributions"]
    assert distributions["available"] is True
    assert distributions["matched"]["pre"]["median"] == pytest.approx(0.80)
    assert distributions["matched"]["post"]["median"] == pytest.approx(0.60)
    assert distributions["split"]["pre"]["median"] == pytest.approx(0.75)
    assert distributions["split"]["post"]["median"] == pytest.approx(0.65)
    assert summary["bootstrap"]["seed"] == 20260830
    assert summary["permutation_seed"] == 20260831


def test_matched_split_summary_does_not_silently_drop_failure_rows() -> None:
    rows, expected = _matched_split_rows()
    rows.loc[0, ["status", "failure_stage", "failure_reason"]] = [
        "failed",
        "score",
        "non-finite logits",
    ]

    with pytest.raises(AnalysisValidationError, match="failed units"):
        summarize_matched_split(
            rows,
            expected_keys=expected,
            key_columns=(
                "model",
                "task",
                "layer",
                "pair_seed",
                "method",
                "condition",
            ),
            expected_blocks=12,
            bootstrap_draws=50,
        )


def test_matched_split_summary_excludes_floor_null_pair_symmetrically() -> None:
    rows, expected = _matched_split_rows()
    split_index = rows.index[
        (rows["model"] == "m0")
        & (rows["task"] == "t0")
        & (rows["layer"] == 1)
        & (rows["pair_seed"] == 0)
        & (rows["condition"] == "split")
    ][0]
    rows.loc[split_index, "target_damage_C"] = np.nan
    rows.loc[
        split_index,
        ["status", "failure_stage", "failure_reason"],
    ] = [
        "pre_target_below_floor",
        "scoring",
        "pre-edit target accuracy is below floor",
    ]

    summary = summarize_matched_split(
        rows,
        expected_keys=expected,
        key_columns=(
            "model",
            "task",
            "layer",
            "pair_seed",
            "method",
            "condition",
        ),
        expected_blocks=12,
        bootstrap_draws=50,
    )

    assert summary["included_units"]["pairs"] == 120
    assert summary["analyzed_units"]["pairs"] == 119
    assert summary["included_units"]["cells"] == 24
    assert summary["primary_score_null_units"] == 1
    assert summary["primary_score_null_pairs"] == 1
    assert summary["failed_units"] == 0


def test_matched_split_summary_rejects_paired_edit_hash_mismatch() -> None:
    rows, expected = _matched_split_rows()
    split_index = rows.index[rows["condition"] == "split"][0]
    rows.loc[split_index, "edit_hash"] = "different-edited-tensor"

    with pytest.raises(AnalysisValidationError, match="edit_hash mismatch"):
        summarize_matched_split(
            rows,
            expected_keys=expected,
            key_columns=(
                "model",
                "task",
                "layer",
                "pair_seed",
                "method",
                "condition",
            ),
            expected_blocks=12,
            bootstrap_draws=50,
        )


def _epsilon_rows() -> tuple[pd.DataFrame, list[tuple[object, ...]]]:
    records: list[dict[str, object]] = []
    expected: list[tuple[object, ...]] = []
    for method in ("fgsm", "pgd"):
        for condition in ("matched", "split"):
            for model, task in (("m0", "sva"), ("m1", "sst2")):
                for epsilon in (0.0, 0.25, 0.5):
                    for pair_seed in (0, 1):
                        if epsilon == 0.0:
                            damage = 0.0
                        elif epsilon == 0.25:
                            damage = 0.95 if condition == "matched" else 0.60
                        elif condition == "matched":
                            damage = 1.0 if pair_seed == 0 else 0.50
                        else:
                            damage = 0.75
                        records.append(
                            {
                                "model": model,
                                "task": task,
                                "layer": 6,
                                "pair_seed": pair_seed,
                                "method": method,
                                "condition": condition,
                                "epsilon": epsilon,
                                "target_damage_C": damage,
                                "control_preservation_S": 0.9,
                                "H": 0.0 if damage == 0 else 0.8,
                                "realized_linf_norm": epsilon,
                                "edit_hash": (
                                    f"{model}:{task}:6:{pair_seed}:{method}:{epsilon}"
                                ),
                                "status": "ok",
                            }
                        )
                        expected.append(
                            (
                                model,
                                task,
                                6,
                                pair_seed,
                                method,
                                condition,
                                epsilon,
                            )
                        )
    return pd.DataFrame(records), expected


def test_epsilon_summary_reports_full_curve_and_zero_noop() -> None:
    rows, expected = _epsilon_rows()

    summary = summarize_epsilon_sweep(
        rows,
        expected_keys=expected,
        key_columns=(
            "model",
            "task",
            "layer",
            "pair_seed",
            "method",
            "condition",
            "epsilon",
        ),
    )

    assert summary["included_units"]["rows"] == 48
    assert summary["included_units"]["model_task_blocks"] == 2
    assert summary["failed_units"] == 0
    assert len(summary["curves"]) == 12
    assert summary["epsilon_zero_integrity"] == {
        "row_count": 16,
        "analyzed_damage_row_count": 16,
        "score_null_row_count": 0,
        "max_abs_target_damage_C": 0.0,
        "max_realized_linf_norm": 0.0,
        "max_abs_target_accuracy_noop_difference": 0.0,
        "passed": True,
    }
    fgsm_matched = next(
        row
        for row in summary["curves"]
        if row["method"] == "fgsm"
        and row["condition"] == "matched"
        and row["epsilon"] == 0.5
    )
    assert fgsm_matched["mean_target_damage_C"] == pytest.approx(0.75)
    # The ceiling fraction is defined over paired experiment rows, not over
    # pair-averaged cells (whose means are both 0.75 here).
    assert fgsm_matched["fraction_at_C_equal_1"] == pytest.approx(0.5)
    assert fgsm_matched["n_pairs"] == 4
    assert summary["has_large_nonmonotonic_reversals"] is True
    assert {
        (row["method"], row["condition"], row["epsilon_from"], row["epsilon_to"])
        for row in summary["large_nonmonotonic_reversals"]
    } == {
        ("fgsm", "matched", 0.25, 0.5),
        ("pgd", "matched", 0.25, 0.5),
    }


def test_epsilon_summary_excludes_floor_null_pair_symmetrically() -> None:
    rows, expected = _epsilon_rows()
    split_index = rows.index[
        (rows["model"] == "m0")
        & (rows["task"] == "sva")
        & (rows["pair_seed"] == 0)
        & (rows["method"] == "fgsm")
        & np.isclose(rows["epsilon"], 0.25)
        & (rows["condition"] == "split")
    ][0]
    rows.loc[split_index, "target_damage_C"] = np.nan
    rows.loc[
        split_index,
        ["status", "failure_stage", "failure_reason"],
    ] = [
        "pre_target_below_floor",
        "scoring",
        "pre-edit target accuracy is below floor",
    ]

    summary = summarize_epsilon_sweep(
        rows,
        expected_keys=expected,
        key_columns=(
            "model",
            "task",
            "layer",
            "pair_seed",
            "method",
            "condition",
            "epsilon",
        ),
    )

    assert summary["included_units"]["rows"] == 48
    assert summary["analyzed_units"]["rows"] == 46
    assert summary["score_null_units"] == 1
    assert summary["score_null_pairs"] == 1
    assert summary["failed_units"] == 0


def test_epsilon_summary_rejects_paired_edit_hash_mismatch() -> None:
    rows, expected = _epsilon_rows()
    split_index = rows.index[rows["condition"] == "split"][0]
    rows.loc[split_index, "edit_hash"] = "different-edited-tensor"

    with pytest.raises(AnalysisValidationError, match="edit_hash mismatch"):
        summarize_epsilon_sweep(
            rows,
            expected_keys=expected,
            key_columns=(
                "model",
                "task",
                "layer",
                "pair_seed",
                "method",
                "condition",
                "epsilon",
            ),
        )


def test_epsilon_summary_uses_nonblank_epsilon_hash_alias() -> None:
    rows, expected = _epsilon_rows()
    rows["epsilon_edit_hash"] = rows["edit_hash"]
    rows["edit_hash"] = None

    summary = summarize_epsilon_sweep(
        rows,
        expected_keys=expected,
        key_columns=(
            "model",
            "task",
            "layer",
            "pair_seed",
            "method",
            "condition",
            "epsilon",
        ),
    )

    assert summary["validated_paired_edit_hashes"] == 24


def _construct_rows() -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    records: list[dict[str, object]] = []
    expected: list[tuple[str, str, str]] = []
    for edit_index, edit_object in enumerate(("alterrep", "linear", "mlp")):
        edit_id = f"edit-{edit_index}"
        for evaluation_family in ("fixed", "fresh_linear", "fresh_mlp"):
            for label in ("target", "control"):
                raw_accuracy = 0.40 if (
                    edit_id == "edit-0"
                    and evaluation_family == "fixed"
                    and label == "target"
                ) else 0.70
                records.append(
                    {
                        "edit_id": edit_id,
                        "edit_object": edit_object,
                        "candidate_architecture": (
                            edit_object if edit_object in {"linear", "mlp"} else None
                        ),
                        "evaluation_family": evaluation_family,
                        "label": label,
                        "accuracy": raw_accuracy,
                        "calibrated_orientation_accuracy": 0.80,
                        "C_raw": 0.60,
                        "C_orientation_calibrated": 0.30,
                        "target_recovery_ratio": 0.70,
                        "control_retention_ratio": 0.90,
                        "status": "ok",
                    }
                )
                expected.append((edit_id, evaluation_family, label))
    return pd.DataFrame(records), expected


def test_construct_summary_covers_every_edit_and_reports_inversion() -> None:
    rows, expected = _construct_rows()

    summary = summarize_construct_check(
        rows,
        expected_keys=expected,
        key_columns=("edit_id", "evaluation_family", "label"),
        expected_edit_ids=("edit-0", "edit-1", "edit-2"),
    )

    assert summary["included_units"] == {"rows": 18, "edits": 3}
    assert summary["failed_units"] == 0
    assert summary["inversion_count"] == 1
    assert summary["redecodable_count"] == 3
    assert summary["redecodable_edit_ids"] == ["edit-0", "edit-1", "edit-2"]
    candidate_summary = summary["candidate_distribution_summary"]
    assert candidate_summary["candidate_edit_count"] == 2
    assert {row["candidate_architecture"] for row in candidate_summary["groups"]} == {
        "linear",
        "mlp",
    }
    for group in candidate_summary["groups"]:
        accuracy = group["metrics"]["accuracy"]
        assert accuracy["median"] in {0.4, 0.7}
        assert accuracy["iqr"] >= 0.0
        assert accuracy["values"] == sorted(accuracy["values"])
    assert {row["evaluation_family"] for row in summary["aggregates"]} == {
        "fixed",
        "fresh_linear",
        "fresh_mlp",
    }


def test_summaries_reject_nonfinite_metrics_instead_of_dropping_them() -> None:
    rows, expected = _epsilon_rows()
    rows.loc[3, "target_damage_C"] = np.nan

    with pytest.raises(AnalysisValidationError, match="non-finite"):
        summarize_epsilon_sweep(
            rows,
            expected_keys=expected,
            key_columns=(
                "model",
                "task",
                "layer",
                "pair_seed",
                "method",
                "condition",
                "epsilon",
            ),
        )
