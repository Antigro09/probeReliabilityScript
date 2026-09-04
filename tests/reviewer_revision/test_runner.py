from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.probes import ProbeTrainConfig
from src.reviewer_revision import runner as runner_module
from src.reviewer_revision.artifacts import RunContext, sha256_file
from src.reviewer_revision.config import load_revision_config
from src.reviewer_revision.experiments import (
    alterrep_edit,
    score_conditions_for_edit,
    train_attacker_evaluator_pair,
)
from src.reviewer_revision.extension_config import ConstructCell
from src.reviewer_revision.runner import (
    MANUSCRIPT_TEMPLATE_PATH,
    MANUSCRIPT_TEMPLATE_TEXT_SHA256,
    WORKSHOP_LIMIT_SOURCE,
    WORKSHOP_MAIN_TEXT_PAGE_LIMIT,
    _augment_matched_artifacts,
    _construct_candidate_device,
    _construct_row_base,
    _decode_pdftotext_output,
    _environment_report,
    _failure_rows,
    _hard_failure_mask,
    _input_manifest,
    _main_text_page_count_from_text,
    _materialize_shards,
    _normalize_score_status,
    _pdfinfo_author_is_anonymous,
    _project_disk_usage,
    _project_runtime_from_benchmarks,
    _regenerate_construct_provenance,
    _requires_padding_fix_regeneration,
    _stage_is_complete,
    _validate_epsilon_baseline_against_matched,
    execute,
    load_pair_checkpoint,
    resolve_resume_directory,
    save_pair_checkpoint,
)
from src.ws5_repaired import EvaluatorQualityError


def test_construct_cell_quality_failure_is_materialized_as_nonestimable(
    monkeypatch, tmp_path
):
    config = load_revision_config("revision_experiment_spec.yaml")
    context = RunContext.create(
        output_root=tmp_path,
        config_hash=config.config_hash,
        git_commit="a" * 40,
        timestamp=datetime(2026, 9, 3, tzinfo=timezone.utc),
    )
    cell = ConstructCell("bert", "google-bert/bert-base-uncased", "sva", 6)

    def raise_blind_evaluator(*args, **kwargs):
        raise EvaluatorQualityError("certifier below locked quality floor")

    monkeypatch.setattr(
        runner_module, "_run_construct_cell", raise_blind_evaluator
    )

    summary = runner_module.run_construct_cell(
        context,
        config,
        cell=cell,
        device=torch.device("cpu"),
    )

    assert summary["status"] == "nonestimable"
    assert summary["failure_stage"] == "evaluator_quality_gate"
    group_rows = pd.read_parquet(
        context.run_dir
        / "construct"
        / "cells"
        / cell.slug
        / "construct_group_rows.parquet"
    )
    assert group_rows["status"].tolist() == ["nonestimable"]
    context.close()


def test_locked_manuscript_template_matches_declared_text_hash():
    template_text = MANUSCRIPT_TEMPLATE_PATH.read_text(encoding="utf-8")

    assert (
        hashlib.sha256(template_text.encode()).hexdigest()
        == MANUSCRIPT_TEMPLATE_TEXT_SHA256
    )
    assert template_text.endswith("\\end{document}\n")


def test_input_manifest_uses_locked_prepatch_manuscript_template():
    manuscript_input = _input_manifest()["main_revised.tex"]

    assert manuscript_input["repository_path"] == (
        "assets/reviewer_revision/main_revised_prepatch.tex"
    )
    assert manuscript_input["repository_copy_sha256"] == sha256_file(
        MANUSCRIPT_TEMPLATE_PATH
    )


def test_score_floor_status_remains_a_scientific_null_not_a_hard_failure():
    row = _normalize_score_status(
        {
            "status": "pre_target_below_floor",
            "reason": "pre-edit target accuracy is below floor",
        }
    )

    assert row["status"] == "pre_target_below_floor"
    assert row["score_status"] == "pre_target_below_floor"
    assert row["failure_stage"] == "scoring"
    assert row["failure_reason"] == "pre-edit target accuracy is below floor"
    statuses = pd.DataFrame(
        [{"status": "ok"}, row, {"status": "failed"}, {"status": "invalid"}]
    )
    assert _hard_failure_mask(statuses).tolist() == [False, False, True, True]


def test_epsilon_baseline_comparison_accepts_matching_score_nulls():
    base = {
        "model_key": "tiny",
        "task": "toy",
        "layer": 1,
        "pair_seed": 0,
        "method": "fgsm",
        "condition": "matched",
        "edit_hash": "same-edit",
        "status": "pre_target_below_floor",
        "C": np.nan,
        "S": np.nan,
        "H": np.nan,
        "target_acc_pre": 0.4,
        "target_acc_post": 0.4,
        "control_acc_pre": 0.8,
        "control_acc_post": 0.8,
    }
    matched = pd.DataFrame([base])
    epsilon = pd.DataFrame(
        [{**base, "epsilon": 0.5, "epsilon_scope": "required_middle"}]
    )

    report = _validate_epsilon_baseline_against_matched(
        matched,
        epsilon,
        expected_rows=1,
    )

    assert report["passed"] is True
    assert report["score_statuses_equal"] is True
    assert report["maximum_absolute_deviations"]["C"] == 0.0


def test_epsilon_baseline_comparison_rejects_asymmetric_score_null():
    base = {
        "model_key": "tiny",
        "task": "toy",
        "layer": 1,
        "pair_seed": 0,
        "method": "fgsm",
        "condition": "matched",
        "edit_hash": "same-edit",
        "status": "pre_target_below_floor",
        "C": np.nan,
        "S": np.nan,
        "H": np.nan,
        "target_acc_pre": 0.4,
        "target_acc_post": 0.4,
        "control_acc_pre": 0.8,
        "control_acc_post": 0.8,
    }
    matched = pd.DataFrame([base])
    epsilon = pd.DataFrame(
        [
            {
                **base,
                "epsilon": 0.5,
                "epsilon_scope": "required_middle",
                "C": 0.0,
            }
        ]
    )

    with pytest.raises(ValueError, match="null pattern"):
        _validate_epsilon_baseline_against_matched(
            matched,
            epsilon,
            expected_rows=1,
        )


def _write_construct_provenance_artifacts(tmp_path):
    construct_dir = tmp_path / "construct"
    construct_dir.mkdir()
    split_payload = {
        "intervention_subdivision_hashes": {
            "direction_fit": "direction-hash",
            "fresh_decoder_fit": "decoder-hash",
            "orientation_calibration": "calibration-hash",
            "final_test": "final-hash",
        },
        "group_disjointness": True,
    }
    baseline_payload = {
        "fresh_linear": {
            "target": {
                "metrics": {"accuracy": 0.75},
                "checkpoint_hashes": ["linear-target-checkpoint"],
            },
            "control": {
                "metrics": {"accuracy": 0.80},
                "checkpoint_hashes": ["linear-control-checkpoint"],
            },
        },
        "fresh_mlp": {
            "target": {
                "metrics": {"accuracy": 0.78},
                "checkpoint_hashes": ["mlp-target-checkpoint"],
            },
            "control": {
                "metrics": {"accuracy": 0.81},
                "checkpoint_hashes": ["mlp-control-checkpoint"],
            },
        },
    }
    hyperparameter_payload = {
        "target": {"linear_selected": {"learning_rate": 0.001}},
        "control": {"linear_selected": {"learning_rate": 0.003}},
    }
    paths = {
        "split": construct_dir / "split_manifest.json",
        "baseline": construct_dir / "fresh_decoder_baselines.json",
        "hyperparameters": construct_dir / "hyperparameter_selection.json",
    }
    for name, payload in (
        ("split", split_payload),
        ("baseline", baseline_payload),
        ("hyperparameters", hyperparameter_payload),
    ):
        paths[name].write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    provenance = {
        "split_manifest_ref": "construct/split_manifest.json",
        "split_manifest_sha256": sha256_file(paths["split"]),
        "split_hashes": split_payload["intervention_subdivision_hashes"],
        "fresh_decoder_baseline_ref": "construct/fresh_decoder_baselines.json",
        "fresh_decoder_baseline_sha256": sha256_file(paths["baseline"]),
        "hyperparameter_selection_ref": "construct/hyperparameter_selection.json",
        "hyperparameter_selection_sha256": sha256_file(paths["hyperparameters"]),
    }
    return paths, provenance, split_payload, baseline_payload


def test_construct_rows_persist_independent_analysis_provenance(tmp_path):
    _, provenance, _, _ = _write_construct_provenance_artifacts(tmp_path)

    row = _construct_row_base(
        context=SimpleNamespace(run_id="test-run"),
        config=SimpleNamespace(config_hash="config-hash"),
        edit_id="dcand_crossfit:linear:0",
        edit_kind="dcand_crossfit",
        architecture="linear",
        seed=0,
        evaluation_family="fresh_linear",
        label="target",
        edit_hash="edit-hash",
        candidate_checkpoint_hash="candidate-hash",
        evaluator_checkpoint_hashes=["evaluator-hash"],
        split_manifest_ref=provenance["split_manifest_ref"],
        artifact_provenance=provenance,
    )

    for field, expected in provenance.items():
        assert row[field] == expected


def test_construct_provenance_regenerates_from_rows_and_hashed_artifacts(tmp_path):
    _, provenance, split_payload, baseline_payload = (
        _write_construct_provenance_artifacts(tmp_path)
    )
    rows = pd.DataFrame(
        [
            {
                "edit_id": "edit-0",
                "evaluation_family": "fixed",
                "label": "target",
                **provenance,
            },
            {
                "edit_id": "edit-0",
                "evaluation_family": "fresh_linear",
                "label": "target",
                "pre_edit_accuracy": 0.75,
                **provenance,
            },
        ]
    )

    regenerated = _regenerate_construct_provenance(
        SimpleNamespace(run_dir=tmp_path), rows
    )

    assert regenerated["split_hashes"] == split_payload[
        "intervention_subdivision_hashes"
    ]
    assert regenerated["group_disjointness"] is True
    assert regenerated["fresh_decoder_unedited_baselines"] == baseline_payload
    assert regenerated["hyperparameter_selection_ref"] == provenance[
        "hyperparameter_selection_ref"
    ]
    assert regenerated["hyperparameter_selection_sha256"] == provenance[
        "hyperparameter_selection_sha256"
    ]
    assert regenerated["referenced_artifact_hashes"] == {
        provenance["split_manifest_ref"]: provenance["split_manifest_sha256"],
        provenance["fresh_decoder_baseline_ref"]: provenance[
            "fresh_decoder_baseline_sha256"
        ],
        provenance["hyperparameter_selection_ref"]: provenance[
            "hyperparameter_selection_sha256"
        ],
    }


def test_construct_provenance_rejects_tampered_referenced_artifact(tmp_path):
    paths, provenance, _, _ = _write_construct_provenance_artifacts(tmp_path)
    rows = pd.DataFrame([{"edit_id": "edit-0", **provenance}])
    paths["hyperparameters"].write_text(
        json.dumps({"tampered": True}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="hyperparameter.*hash mismatch"):
        _regenerate_construct_provenance(SimpleNamespace(run_dir=tmp_path), rows)


def test_construct_provenance_rejects_row_baseline_mismatch(tmp_path):
    _, provenance, _, _ = _write_construct_provenance_artifacts(tmp_path)
    rows = pd.DataFrame(
        [
            {
                "edit_id": "edit-0",
                "evaluation_family": "fresh_linear",
                "label": "target",
                "pre_edit_accuracy": 0.99,
                **provenance,
            }
        ]
    )

    with pytest.raises(RuntimeError, match="fresh baseline accuracy mismatch"):
        _regenerate_construct_provenance(SimpleNamespace(run_dir=tmp_path), rows)


def _tiny_pair():
    generator = torch.Generator().manual_seed(12)
    X = torch.randn(80, 4, generator=generator)
    zc = (X[:, 0] > 0).long()
    ze = (X[:, 1] > 0).long()
    return train_attacker_evaluator_pair(
        X[:40], zc[:40], ze[:40], X[40:], zc[40:], ze[40:],
        pair_seed=2,
        device=torch.device("cpu"),
        config=ProbeTrainConfig(epochs=4, batch_size=20),
    ), X


def test_pair_checkpoint_roundtrip_preserves_hashes_and_logits(tmp_path):
    pair, X = _tiny_pair()
    path = tmp_path / "pair.pt"
    save_pair_checkpoint(path, pair, metadata={"cell": ["tiny", "toy", 1]})
    loaded, metadata = load_pair_checkpoint(path, device=torch.device("cpu"))

    assert metadata["cell"] == ["tiny", "toy", 1]
    assert loaded.attacker.target_checkpoint_hash == pair.attacker.target_checkpoint_hash
    assert loaded.attacker.control_checkpoint_hash == pair.attacker.control_checkpoint_hash
    assert loaded.evaluator.target_checkpoint_hash == pair.evaluator.target_checkpoint_hash
    assert loaded.evaluator.control_checkpoint_hash == pair.evaluator.control_checkpoint_hash
    assert torch.equal(
        loaded.attacker.target_probe(X), pair.attacker.target_probe(X)
    )


def test_matched_alterrep_artifacts_are_losslessly_reconstructible(tmp_path):
    pair, X = _tiny_pair()
    X_score = X[40:]
    target = (X_score[:, 0] > 0).long()
    control = (X_score[:, 1] > 0).long()
    edited = alterrep_edit(
        X_score,
        target,
        pair.attacker.target_probe,
        device=torch.device("cpu"),
    )
    rows = score_conditions_for_edit(
        X_pre=X_score,
        X_post=edited,
        target_labels=target,
        control_labels=control,
        matched_target_probe=pair.attacker.target_probe,
        matched_control_probe=pair.attacker.control_probe,
        split_target_probe=pair.evaluator.target_probe,
        split_control_probe=pair.evaluator.control_probe,
        device=torch.device("cpu"),
        common={
            "model_key": "tiny",
            "task": "toy",
            "layer": 1,
            "pair_seed": 2,
            "method": "alterrep",
        },
    )
    context = RunContext.create(
        output_root=tmp_path,
        config_hash="a" * 64,
        git_commit="b" * 40,
        timestamp=datetime(2026, 8, 29, 9, tzinfo=timezone.utc),
    )
    try:
        _augment_matched_artifacts(
            context,
            rows,
            pair=pair,
            X_pre=X_score,
            X_post=edited,
            target_labels=target,
            control_labels=control,
            example_ids=[f"example-{index}" for index in range(len(X_score))],
            source_indices=list(range(40, 80)),
            split_name="score",
            source_cache_sha256="c" * 64,
            source_data_hash="d" * 64,
            method="alterrep",
            device=torch.device("cpu"),
        )
        recipe_path = context.run_dir / rows[0]["edit_artifact_ref"]
        recipe = torch.load(recipe_path, map_location="cpu", weights_only=True)
        signs = torch.where(target == 1, -1.0, 1.0).unsqueeze(1)
        reconstructed = X_score.float() + signs * recipe["direction"].unsqueeze(0)
        assert torch.equal(reconstructed, edited.float())
        assert rows[0]["edit_artifact_ref"] == rows[1]["edit_artifact_ref"]
        assert rows[0]["per_example_artifact_ref"] != rows[1][
            "per_example_artifact_ref"
        ]
    finally:
        context.close()


def test_resume_directory_selects_latest_matching_unlocked_run(tmp_path):
    config_hash = "a" * 64
    commit = "b" * 40
    first = RunContext.create(
        output_root=tmp_path,
        config_hash=config_hash,
        git_commit=commit,
        timestamp=datetime(2026, 8, 29, 10, tzinfo=timezone.utc),
    )
    first.close()
    second = RunContext.create(
        output_root=tmp_path,
        config_hash=config_hash,
        git_commit=commit,
        timestamp=datetime(2026, 8, 29, 11, tzinfo=timezone.utc),
    )
    second.close()

    assert resolve_resume_directory(tmp_path, config_hash=config_hash) == second.run_dir
    assert resolve_resume_directory(second.run_dir, config_hash=config_hash) == second.run_dir


def test_resume_directory_recovers_a_stale_dead_process_lock(tmp_path):
    config_hash = "c" * 64
    context = RunContext.create(
        output_root=tmp_path,
        config_hash=config_hash,
        git_commit="d" * 40,
        timestamp=datetime(2026, 8, 29, 12, tzinfo=timezone.utc),
    )
    run_dir = context.run_dir
    context.close()
    (run_dir / ".run.lock").write_text(
        json.dumps({"pid": 2_147_483_647, "token": "stale"}), encoding="utf-8"
    )

    assert resolve_resume_directory(tmp_path, config_hash=config_hash) == run_dir
    assert not (run_dir / ".run.lock").exists()
    assert list(run_dir.glob(".stale-run-lock-*.json"))


def test_completed_baseline_gate_revalidates_current_rerun_hash(tmp_path):
    from src.reviewer_revision.runner import ARCHIVE_PATH

    config = load_revision_config("revision_experiment_spec.yaml")
    context = RunContext.create(
        output_root=tmp_path,
        config_hash=config.config_hash,
        git_commit="d" * 40,
        timestamp=datetime(2026, 8, 29, 12, 30, tzinfo=timezone.utc),
    )
    try:
        current_path = context.run_dir / "baseline_current_12_cells.json"
        current_path.write_text(json.dumps({"cells": {}}), encoding="utf-8")
        report = {
            "status": "ok",
            "archive_sha256": sha256_file(ARCHIVE_PATH),
            "current_rerun_ref": current_path.name,
            "current_rerun_sha256": sha256_file(current_path),
        }
        (context.run_dir / "baseline_reproduction.json").write_text(
            json.dumps(report), encoding="utf-8"
        )

        assert _stage_is_complete(context, "reproduce-baseline", config=config)
        current_path.write_text(json.dumps({"tampered": True}), encoding="utf-8")
        with pytest.raises(RuntimeError, match="rerun artifact hash mismatch"):
            _stage_is_complete(context, "reproduce-baseline", config=config)
    finally:
        context.close()


def test_execute_all_dispatches_locked_stages_in_order_and_records_gate(monkeypatch, tmp_path):
    from src.reviewer_revision import runner

    config = load_revision_config("revision_experiment_spec.yaml")
    observed: list[str] = []
    stage_results = {
        "preflight": {"status": "ok"},
        "benchmark": {"status": "ok"},
        "reproduce-baseline": {"status": "ok"},
        "matched-split": {"status": "ok"},
        "epsilon-sweep": {"status": "ok"},
        "construct-check": {"status": "ok"},
        "analyze": {"status": "complete", "validation": {"complete": True}},
        "figures": {"status": "ok"},
        "patch-paper": {"status": "compiled_pending_visual_inspection"},
    }
    functions = {
        "preflight": "run_preflight",
        "benchmark": "run_benchmark",
        "reproduce-baseline": "run_reproduce_baseline",
        "matched-split": "run_matched_split",
        "epsilon-sweep": "run_epsilon_sweep",
        "construct-check": "run_construct_check",
        "analyze": "run_analysis",
        "figures": "run_figures",
        "patch-paper": "run_patch_paper",
    }

    def replacement(stage):
        def run(*_args, **_kwargs):
            observed.append(stage)
            return stage_results[stage]

        return run

    for stage, function_name in functions.items():
        monkeypatch.setattr(runner, function_name, replacement(stage))
    monkeypatch.setattr(runner, "_current_commit", lambda: "e" * 40)
    monkeypatch.setattr(runner, "_starting_commit", lambda: "f" * 40)
    monkeypatch.setattr(runner, "_input_manifest", dict)
    args = SimpleNamespace(
        output_root=tmp_path,
        resume=False,
        device="cpu",
        config=config.source_path,
        log_level="INFO",
    )

    assert execute("all", config, args) == 0
    assert observed == list(stage_results)
    manifest_path = next(tmp_path.glob("*/run_manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["pipeline"]["status"] == "pending_visual_inspection"
    assert manifest["pipeline"]["last_completed_stage"] == "patch-paper"


def test_runtime_projection_counts_every_epsilon_attack_and_reference_control():
    benchmark = {
        "cache_load_seconds": 2.0,
        "probe_training_seconds": 8.0,
        "coupled_pair_seconds": 6.0,
        "reference_generation_seconds": 4.0,
        "reference_pair_scoring_seconds": 2.0,
        "serialization_seconds": 1.0,
        "method_timings": {
            "fgsm": {"generation_seconds": 1.0, "both_condition_scores_seconds": 0.5},
            "pgd": {"generation_seconds": 2.0, "both_condition_scores_seconds": 0.5},
        },
    }
    projection = _project_runtime_from_benchmarks(
        [benchmark], selected_cells=60, pair_seeds=5
    )

    assert projection["full_grid_pair_units"] == 300
    assert projection["fallback_pair_units"] == 180
    assert projection["required_middle_epsilon_pair_units"] == 60
    assert projection["required_middle_epsilon_attack_invocations"] == 1200
    assert projection["optional_all_layer_epsilon_pair_units"] == 300
    assert projection["full_grid_seconds"] == pytest.approx(
        60 * (2.0 + 4.0) + 300 * (8.0 + 6.0 + 2.0 + 1.0)
    )
    assert projection["required_middle_epsilon_seconds"] == pytest.approx(
        60 * (10 * (1.0 + 0.5 + 2.0 + 0.5) + 20 * 1.0)
    )


def test_environment_report_normalizes_project_path_for_psutil(monkeypatch):
    from src.reviewer_revision import runner

    observed: list[str] = []
    real_disk_usage = runner.psutil.disk_usage

    def strict_disk_usage(path):
        assert isinstance(path, str)
        observed.append(path)
        return real_disk_usage(path)

    monkeypatch.setattr(runner.psutil, "disk_usage", strict_disk_usage)
    report = _environment_report(torch.device("cpu"))

    assert observed
    assert report["selected_device"] == "cpu"
    assert report["device_protocol"] == {
        "transformer_extraction": "cpu",
        "candidate_mlp_and_jacobian": "cpu",
        "candidate_linear_and_mka": "cpu",
        "linear_attackers_and_evaluators": "cpu",
        "fresh_decoders": "cpu",
        "statistics": "cpu",
    }
    assert report["disk_free_bytes"] > 0


def test_construct_device_routing_accelerates_only_mlp_candidate_work():
    requested = torch.device("mps")

    assert _construct_candidate_device("mlp", requested) == requested
    assert _construct_candidate_device("linear", requested) == torch.device("cpu")
    assert _construct_candidate_device("mka", requested) == torch.device("cpu")


def test_disk_projection_exposes_exact_per_cell_accounting():
    cells = [
        SimpleNamespace(model_key="bert", task="sva", layer=6),
        SimpleNamespace(model_key="qwen", task="sst2", layer=14),
    ]
    hidden_sizes = {
        ("bert", "sva", 6): 768,
        ("qwen", "sst2", 14): 1536,
    }
    score_sizes = {"sva": 40, "sst2": 50}

    projection = _project_disk_usage(
        matched_cells=cells,
        epsilon_cells=cells[:1],
        hidden_sizes=hidden_sizes,
        score_sizes=score_sizes,
        pair_seeds=5,
        epsilon_count=10,
        sample_bytes=1_000,
        construct_reserve_bytes=8_000,
    )

    assert len(projection["matched_cells"]) == 2
    assert len(projection["epsilon_cells"]) == 1
    epsilon = projection["epsilon_cells"][0]
    assert epsilon["n_score"] == 40
    assert epsilon["hidden_size"] == 768
    assert epsilon["edit_units"] == 5 * 2 * 10
    assert epsilon["raw_bytes"] == 5 * 2 * 10 * 40 * (4 * 768 + 224)
    raw = sum(row["raw_bytes"] for row in projection["matched_cells"])
    raw += sum(row["raw_bytes"] for row in projection["epsilon_cells"])
    assert projection["raw_artifact_bytes"] == raw
    assert projection["artifact_bytes_with_overhead"] == int(raw * 1.10)
    assert projection["total_bytes"] == int(raw * 1.10) + 8_000


def test_failed_shards_materialize_for_audit_but_remain_retryable(tmp_path):
    key = ("tiny", "toy", 1, 0, "alterrep", "matched")
    context = RunContext.create(
        output_root=tmp_path,
        config_hash="a" * 64,
        git_commit="b" * 40,
        timestamp=datetime(2026, 8, 29, 15, tzinfo=timezone.utc),
    )
    try:
        row = _failure_rows(
            {"model_key": "tiny", "task": "toy", "layer": 1, "pair_seed": 0,
             "method": "alterrep"},
            conditions=("matched",),
            error=RuntimeError("deliberate"),
            failure_stage="edit_generation",
            pre_metrics={"matched": {"target_acc_pre": 0.9, "control_acc_pre": 0.8}},
        )[0]
        context.write_json_shard("matched_split", key, row)
        frame = _materialize_shards(
            context,
            experiment="matched_split",
            expected_keys=(key,),
            csv_name="failed.csv",
            parquet_name="failed.parquet",
            key_columns=(
                "model_key", "task", "layer", "pair_seed", "method", "condition"
            ),
        )
        assert frame.iloc[0]["failure_stage"] == "edit_generation"
        assert frame.iloc[0]["target_acc_pre"] == pytest.approx(0.9)
        assert context.completed_keys("matched_split", expected_keys=(key,)) == set()
        assert (context.run_dir / "failed.parquet").is_file()
    finally:
        context.close()


def test_workshop_main_text_page_count_stops_before_references():
    pages = ["Title\nIntroduction", "Methods", "Conclusion", "References\n[1]", "Appendix"]
    assert _main_text_page_count_from_text(pages) == 3
    with pytest.raises(ValueError, match="References"):
        _main_text_page_count_from_text(["Title", "Methods"])


def test_workshop_page_gate_targets_lp4fm_full_paper_limit():
    assert WORKSHOP_MAIN_TEXT_PAGE_LIMIT == 9
    assert WORKSHOP_LIMIT_SOURCE == "https://lp4fm.github.io/"


@pytest.mark.parametrize(
    ("author_line", "expected"),
    [
        ("Author:          ", True),
        ("Author: Anonymous Authors", True),
        ("Author: Anonymous Author(s)", True),
        ("Author: Named Researcher", False),
    ],
)
def test_pdfinfo_author_gate_accepts_blank_or_anonymous_metadata(
    author_line: str,
    expected: bool,
):
    info = f"Title: Paper\n{author_line}\nPages: 9\n"

    assert _pdfinfo_author_is_anonymous(info) is expected


def test_pdftotext_output_is_decoded_as_utf8_not_windows_codepage():
    encoded = "candidate-conditioned — ε sweep".encode()

    assert _decode_pdftotext_output(encoded) == "candidate-conditioned — ε sweep"


def test_only_legacy_left_padding_cache_requires_padding_fix_regeneration():
    legacy = SimpleNamespace(selection=SimpleNamespace(provenance={}))
    corrected = SimpleNamespace(
        selection=SimpleNamespace(
            provenance={
                "extraction_code_version": "last-nonpadding-mask-index-v2"
            }
        )
    )
    assert _requires_padding_fix_regeneration("gemma", legacy) is True
    assert _requires_padding_fix_regeneration("gemma", corrected) is False
    assert _requires_padding_fix_regeneration("qwen", legacy) is False


def test_execute_records_failure_and_tees_stdout_and_stderr(monkeypatch, tmp_path):
    from src.reviewer_revision import runner

    config = load_revision_config("revision_experiment_spec.yaml")

    def fail(*_args, **_kwargs):
        print("captured standard output")
        print("captured standard error", file=sys.stderr)
        raise RuntimeError("deliberate stage failure")

    monkeypatch.setattr(runner, "run_preflight", fail)
    monkeypatch.setattr(runner, "_current_commit", lambda: "e" * 40)
    monkeypatch.setattr(runner, "_starting_commit", lambda: "f" * 40)
    monkeypatch.setattr(runner, "_input_manifest", dict)
    args = SimpleNamespace(
        output_root=tmp_path,
        resume=False,
        device="cpu",
        config=config.source_path,
        log_level="INFO",
    )

    with pytest.raises(RuntimeError, match="deliberate"):
        execute("preflight", config, args)

    run_dir = next(path.parent for path in tmp_path.glob("*/run_manifest.json"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["pipeline"]["status"] == "failed"
    assert manifest["pipeline"]["error_type"] == "RuntimeError"
    log = (run_dir / "console.log").read_text(encoding="utf-8")
    assert "captured standard output" in log
    assert "captured standard error" in log
    assert "deliberate stage failure" in log
