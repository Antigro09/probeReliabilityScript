from __future__ import annotations

import pandas as pd
import pytest

from src.reviewer_revision.extension_analysis import (
    LOCKED_CONSTRUCT_CANDIDATE_EDIT_IDS,
    AnalysisValidationError,
    classify_construct_panel,
    holm_adjust,
)
from src.reviewer_revision.extension_config import load_extension_config


def _edit_parts(edit_id: str) -> tuple[str, int]:
    prefix, seed_text = edit_id.rsplit("-seed", 1)
    return prefix.removeprefix("dcand_crossfit-"), int(seed_text)


def _panel_rows() -> pd.DataFrame:
    config = load_extension_config("revision_caveat_extension_spec.yaml")
    rows: list[dict[str, object]] = []
    for cell in config.all_cells:
        common = {
            "model_key": cell.model_key,
            "model_id": cell.model_id,
            "task": cell.task,
            "layer": cell.layer,
            "evaluation_family": "fresh_linear",
            "decoder_seed": 0,
            "split_manifest_sha256": "a" * 64,
            "status": "ok",
            "failure_stage": None,
            "failure_reason": None,
        }
        for label in ("target", "control"):
            for group_id in ("g1", "g2"):
                rows.append(
                    {
                        **common,
                        "row_kind": "baseline",
                        "label": label,
                        "group_id": group_id,
                        "n_examples": 10,
                        "correct_count": 9,
                        "decoder_checkpoint_sha256": "b" * 64,
                        "edit_id": None,
                        "edit_object": None,
                        "architecture": None,
                        "candidate_seed": None,
                        "edit_hash": None,
                    }
                )
        for edit_id in LOCKED_CONSTRUCT_CANDIDATE_EDIT_IDS:
            architecture, candidate_seed = _edit_parts(edit_id)
            for label in ("target", "control"):
                for group_id in ("g1", "g2"):
                    rows.append(
                        {
                            **common,
                            "row_kind": "post_edit",
                            "label": label,
                            "group_id": group_id,
                            "n_examples": 10,
                            "correct_count": 9,
                            "decoder_checkpoint_sha256": "c" * 64,
                            "edit_id": edit_id,
                            "edit_object": "dcand_crossfit",
                            "architecture": architecture,
                            "candidate_seed": candidate_seed,
                            "edit_hash": f"{candidate_seed:064x}",
                        }
                    )
    return pd.DataFrame(rows)


def _mark_cell_nonestimable(
    rows: pd.DataFrame, *, model_key: str, task: str
) -> pd.DataFrame:
    mask = (rows["model_key"] == model_key) & (rows["task"] == task)
    selected = rows.loc[mask].iloc[0]
    retained = rows.loc[~mask].copy()
    status_row = {column: None for column in rows.columns}
    status_row.update(
        {
            "model_key": selected["model_key"],
            "model_id": selected["model_id"],
            "task": selected["task"],
            "layer": int(selected["layer"]),
            "row_kind": "cell_status",
            "evaluation_family": "fresh_linear",
            "decoder_seed": 0,
            "n_examples": 0,
            "correct_count": 0,
            "status": "nonestimable",
            "failure_stage": "construct_cell",
            "failure_reason": "explicit test fixture nonestimability",
        }
    )
    return pd.concat([retained, pd.DataFrame([status_row])], ignore_index=True)


def test_construct_panel_uses_counts_and_locked_intersection_union_inference():
    config = load_extension_config("revision_caveat_extension_spec.yaml")
    summary = classify_construct_panel(_panel_rows(), config=config)

    assert summary["status"] == "ok"
    assert summary["confirmatory_cell_count"] == 11
    assert summary["multiplicity"] == {
        "method": "holm_one_sided",
        "family_size": 11,
        "alpha": 0.05,
    }
    assert summary["lower_bounds"] == {
        "confidence_level": 0.95,
        "quantile": 0.05,
        "quantile_method": "linear",
        "scope": "marginal",
        "multiplicity_adjusted": False,
        "simultaneous": False,
    }
    pilot = summary["pilot"]
    assert pilot["confirmatory"] is False
    assert pilot["inference_mode"] == "descriptive_only"
    assert "internal_cell_p_value" not in pilot
    assert "holm_adjusted_cell_p_value" not in pilot
    assert "passes_locked_confirmatory_rule" not in pilot

    cell = summary["cells"]["bert-sva-l6"]
    assert cell["status"] == "ok"
    assert cell["endpoints"] == {
        "median_target_post_edit_accuracy": pytest.approx(0.9),
        "median_target_recovery_ratio": pytest.approx(1.0),
        "median_control_retention_ratio": pytest.approx(1.0),
    }
    assert cell["passes_locked_point_thresholds"] is True
    assert cell["passes_holm_adjusted_inference"] is True
    assert cell["passes_locked_confirmatory_rule"] is True
    assert cell["internal_cell_p_value"] == pytest.approx(1 / 10_001)
    assert cell["holm_adjusted_cell_p_value"] == pytest.approx(11 / 10_001)
    assert set(cell["endpoint_p_values"]) == {
        "accuracy",
        "target_recovery_ratio",
        "control_retention_ratio",
    }


def test_construct_panel_is_deterministic_and_shares_resamples_within_task():
    config = load_extension_config("revision_caveat_extension_spec.yaml")
    rows = _panel_rows()
    first = classify_construct_panel(rows, config=config)
    second = classify_construct_panel(rows, config=config)

    assert first == second
    assert first["bootstrap"]["bit_generator"] == "PCG64"
    assert first["bootstrap"]["draws"] == 10_000
    task_hashes = first["bootstrap"]["task_resample_sha256"]
    assert set(task_hashes) == {"sva", "sst2"}
    assert task_hashes["sva"] != task_hashes["sst2"]
    for cell in config.confirmatory_cells:
        assert (
            first["cells"][cell.slug]["task_resample_sha256"] == task_hashes[cell.task]
        )


def test_construct_panel_keeps_point_threshold_miss_as_successful_analysis():
    config = load_extension_config("revision_caveat_extension_spec.yaml")
    rows = _panel_rows()
    mask = (
        (rows["model_key"] == "bert")
        & (rows["task"] == "sva")
        & (rows["row_kind"] == "post_edit")
        & (rows["label"] == "target")
    )
    rows.loc[mask, "correct_count"] = 5

    summary = classify_construct_panel(rows, config=config)
    cell = summary["cells"]["bert-sva-l6"]

    assert summary["status"] == "ok"
    assert cell["status"] == "ok"
    assert cell["passes_locked_point_thresholds"] is False
    assert cell["passes_holm_adjusted_inference"] is False
    assert cell["passes_locked_confirmatory_rule"] is False


def test_nonestimable_confirmatory_cell_remains_in_fixed_holm_family():
    config = load_extension_config("revision_caveat_extension_spec.yaml")
    rows = _mark_cell_nonestimable(_panel_rows(), model_key="llama", task="sst2")

    summary = classify_construct_panel(rows, config=config)
    cell = summary["cells"]["llama-sst2-l14"]

    assert summary["multiplicity"]["family_size"] == 11
    assert cell["status"] == "nonestimable"
    assert cell["internal_cell_p_value"] == 1.0
    assert cell["holm_adjusted_cell_p_value"] == 1.0
    assert cell["passes_locked_point_thresholds"] is False
    assert cell["passes_holm_adjusted_inference"] is False
    assert cell["passes_locked_confirmatory_rule"] is False


def test_construct_panel_rejects_misaligned_task_groups_and_duplicate_keys():
    config = load_extension_config("revision_caveat_extension_spec.yaml")
    rows = _panel_rows()
    mask = (rows["model_key"] == "bert") & (rows["task"] == "sva")
    rows.loc[mask & (rows["group_id"] == "g2"), "group_id"] = "other"
    with pytest.raises(AnalysisValidationError, match="shared group universe"):
        classify_construct_panel(rows, config=config)

    rows = _panel_rows()
    duplicate = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    with pytest.raises(
        AnalysisValidationError, match="duplicate construct group-row key"
    ):
        classify_construct_panel(duplicate, config=config)


def test_construct_panel_requires_exact_sixty_candidate_edit_identities():
    config = load_extension_config("revision_caveat_extension_spec.yaml")
    rows = _panel_rows()
    rows = rows.loc[rows["edit_id"] != LOCKED_CONSTRUCT_CANDIDATE_EDIT_IDS[-1]].copy()

    with pytest.raises(AnalysisValidationError, match="exact 60 candidate edits"):
        classify_construct_panel(rows, config=config)


def test_construct_panel_requires_explicit_cell_status_not_silent_absence():
    config = load_extension_config("revision_caveat_extension_spec.yaml")
    rows = _panel_rows()
    rows = rows.loc[~((rows["model_key"] == "llama") & (rows["task"] == "sst2"))].copy()

    with pytest.raises(AnalysisValidationError, match="exact cell coverage"):
        classify_construct_panel(rows, config=config)


def test_construct_panel_rejects_mixed_decoder_provenance_and_coercive_counts():
    config = load_extension_config("revision_caveat_extension_spec.yaml")
    rows = _panel_rows()
    mask = (
        (rows["model_key"] == "bert")
        & (rows["task"] == "sva")
        & (rows["row_kind"] == "post_edit")
        & (rows["label"] == "target")
        & (rows["group_id"] == "g1")
        & (rows["edit_id"] == LOCKED_CONSTRUCT_CANDIDATE_EDIT_IDS[0])
    )
    rows.loc[mask, "decoder_checkpoint_sha256"] = "d" * 64
    with pytest.raises(AnalysisValidationError, match="decoder checkpoint hash"):
        classify_construct_panel(rows, config=config)

    rows = _panel_rows()
    rows["n_examples"] = rows["n_examples"].astype(object)
    rows.loc[rows.index[0], "n_examples"] = "10"
    with pytest.raises(
        AnalysisValidationError, match="n_examples must contain integers"
    ):
        classify_construct_panel(rows, config=config)

    rows = _panel_rows()
    rows["correct_count"] = rows["correct_count"].astype(object)
    rows.loc[rows.index[0], "correct_count"] = 9.0
    with pytest.raises(
        AnalysisValidationError, match="correct_count must contain integers"
    ):
        classify_construct_panel(rows, config=config)


def test_construct_points_are_example_weighted_and_ratios_are_not_clipped():
    config = load_extension_config("revision_caveat_extension_spec.yaml")
    rows = _panel_rows()
    cell = rows["task"] == "sva"
    rows.loc[cell & (rows["group_id"] == "g1"), "n_examples"] = 1
    rows.loc[cell & (rows["group_id"] == "g2"), "n_examples"] = 9
    baseline = cell & (rows["row_kind"] == "baseline")
    rows.loc[baseline & (rows["group_id"] == "g1"), "correct_count"] = 1
    rows.loc[baseline & (rows["group_id"] == "g2"), "correct_count"] = 8
    first_half = set(LOCKED_CONSTRUCT_CANDIDATE_EDIT_IDS[:30])
    post = cell & (rows["row_kind"] == "post_edit")
    rows.loc[post & (rows["group_id"] == "g1"), "correct_count"] = 0
    rows.loc[
        post & rows["edit_id"].isin(first_half) & (rows["group_id"] == "g2"),
        "correct_count",
    ] = 3
    rows.loc[
        post & ~rows["edit_id"].isin(first_half) & (rows["group_id"] == "g2"),
        "correct_count",
    ] = 9

    summary = classify_construct_panel(rows, config=config)
    endpoints = summary["cells"]["bert-sva-l6"]["endpoints"]

    assert endpoints["median_target_post_edit_accuracy"] == pytest.approx(0.6)
    assert endpoints["median_target_recovery_ratio"] == pytest.approx(0.25)
    assert endpoints["median_control_retention_ratio"] == pytest.approx(0.25)


def test_bootstrap_chance_denominator_draws_are_conservative_tail_failures():
    config = load_extension_config("revision_caveat_extension_spec.yaml")
    rows = _panel_rows()
    baseline_target = (
        (rows["model_key"] == "bert")
        & (rows["task"] == "sva")
        & (rows["row_kind"] == "baseline")
        & (rows["label"] == "target")
    )
    rows.loc[baseline_target & (rows["group_id"] == "g1"), "correct_count"] = 4

    summary = classify_construct_panel(rows, config=config)
    cell = summary["cells"]["bert-sva-l6"]
    invalid = cell["invalid_bootstrap_draws"]["target_baseline_at_or_below_chance"]

    assert 2_000 < invalid < 3_000
    assert cell["endpoint_p_values"]["target_recovery_ratio"] == pytest.approx(
        (invalid + 1) / 10_001
    )
    assert cell["marginal_lower_bounds"]["target_recovery_ratio"] is None
    assert cell["marginal_lower_bound_finite"]["target_recovery_ratio"] is False


def test_endpoint_tail_uses_less_than_or_equal_boundary():
    config = load_extension_config("revision_caveat_extension_spec.yaml")
    rows = _panel_rows()
    cell = rows["task"] == "sva"
    rows.loc[cell, "n_examples"] = 20
    baseline = cell & (rows["row_kind"] == "baseline")
    rows.loc[baseline, "correct_count"] = 18
    post_target = cell & (rows["row_kind"] == "post_edit") & (rows["label"] == "target")
    rows.loc[post_target, "correct_count"] = 11
    post_control = (
        cell & (rows["row_kind"] == "post_edit") & (rows["label"] == "control")
    )
    rows.loc[post_control, "correct_count"] = 18

    summary = classify_construct_panel(rows, config=config)
    record = summary["cells"]["bert-sva-l6"]

    assert record["endpoints"]["median_target_post_edit_accuracy"] == 0.55
    assert record["endpoint_p_values"]["accuracy"] == 1.0


def test_task_specific_streams_and_confirmatory_results_ignore_pilot_or_other_task():
    config = load_extension_config("revision_caveat_extension_spec.yaml")
    rows = _panel_rows()
    baseline = classify_construct_panel(rows, config=config)

    pilot_changed = rows.copy()
    pilot_post = (
        (pilot_changed["model_key"] == config.pilot.model_key)
        & (pilot_changed["task"] == config.pilot.task)
        & (pilot_changed["row_kind"] == "post_edit")
    )
    pilot_changed.loc[pilot_post, "correct_count"] = 5
    changed = classify_construct_panel(pilot_changed, config=config)
    for cell in config.confirmatory_cells:
        assert changed["cells"][cell.slug] == baseline["cells"][cell.slug]

    no_sva = rows
    for cell in config.confirmatory_cells:
        if cell.task == "sva":
            no_sva = _mark_cell_nonestimable(
                no_sva, model_key=cell.model_key, task=cell.task
            )
    without_sva = classify_construct_panel(no_sva, config=config)
    assert (
        without_sva["bootstrap"]["task_resample_sha256"]["sst2"]
        == baseline["bootstrap"]["task_resample_sha256"]["sst2"]
    )


def test_holm_adjustment_is_step_down_monotone_for_varied_p_values():
    assert holm_adjust({"a": 0.001, "b": 0.02, "c": 0.5}) == pytest.approx(
        {"a": 0.003, "b": 0.04, "c": 0.5}
    )
