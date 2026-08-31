from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pandas as pd
import pytest
import torch

from src.reviewer_revision import runner
from src.reviewer_revision.artifacts import RunContext, sha256_json, sha256_tensor
from src.reviewer_revision.config import ConstructEditKey
from src.reviewer_revision.extension_config import ConstructCell
from src.reviewer_revision.runner import (
    _construct_layout,
    _construct_row_base,
    construct_artifact_root,
    construct_row_identity,
    run_construct_cell,
    run_construct_check,
)
from src.reviewer_revision.training import DecoderSpec, HyperparameterSelection


def _cells() -> tuple[ConstructCell, ConstructCell]:
    return (
        ConstructCell("bert", "google-bert/bert-base-uncased", "sva", 6),
        ConstructCell("qwen", "Qwen/Qwen2.5-1.5B", "sst2", 14),
    )


def test_construct_cell_layout_namespaces_every_artifact_family(tmp_path):
    bert, qwen = _cells()

    bert_layout = _construct_layout(tmp_path, bert)
    qwen_layout = _construct_layout(tmp_path, qwen)

    assert construct_artifact_root(tmp_path, bert) == (
        tmp_path / "construct" / "cells" / "bert-sva-l6"
    )
    assert construct_artifact_root(tmp_path, qwen) == (
        tmp_path / "construct" / "cells" / "qwen-sst2-l14"
    )
    for left, right in (
        (bert_layout.artifact_root, qwen_layout.artifact_root),
        (bert_layout.checkpoint_root, qwen_layout.checkpoint_root),
        (bert_layout.array_root, qwen_layout.array_root),
        (bert_layout.rows_csv, qwen_layout.rows_csv),
        (bert_layout.rows_parquet, qwen_layout.rows_parquet),
        (bert_layout.summary_path, qwen_layout.summary_path),
    ):
        assert left != right
    assert bert_layout.checkpoint_root == (
        tmp_path / "checkpoints" / "construct" / "cells" / bert.slug
    )
    assert bert_layout.array_root == (
        tmp_path / "arrays" / "construct" / "cells" / bert.slug
    )
    assert bert_layout.summary_cell == {
        "model_key": bert.model_key,
        "model_id": bert.model_id,
        "task": bert.task,
        "layer": bert.layer,
    }


def test_construct_cell_shards_include_full_identity_and_cannot_collide(tmp_path):
    bert, qwen = _cells()
    edit_key = ("dcand_crossfit", "linear", 3)

    bert_layout = _construct_layout(tmp_path, bert)
    qwen_layout = _construct_layout(tmp_path, qwen)
    bert_key = bert_layout.shard_key(edit_key)
    qwen_key = qwen_layout.shard_key(edit_key)

    assert bert_layout.shard_experiment == "construct_edits.bert-sva-l6"
    assert qwen_layout.shard_experiment == "construct_edits.qwen-sst2-l14"
    assert bert_layout.shard_experiment != qwen_layout.shard_experiment
    assert bert_key[:4] == (
        bert.model_key,
        bert.model_id,
        bert.task,
        bert.layer,
    )
    assert qwen_key[:4] == (
        qwen.model_key,
        qwen.model_id,
        qwen.task,
        qwen.layer,
    )
    assert bert_key != qwen_key


def test_construct_rows_carry_full_cell_identity():
    bert, qwen = _cells()
    common = {
        "context": SimpleNamespace(run_id="test-run"),
        "config": SimpleNamespace(config_hash="config-hash"),
        "edit_id": "dcand_crossfit-linear-seed3",
        "edit_kind": "dcand_crossfit",
        "architecture": "linear",
        "seed": 3,
        "evaluation_family": "fixed",
        "label": "target",
        "edit_hash": "edit-hash",
        "candidate_checkpoint_hash": "candidate-hash",
        "evaluator_checkpoint_hashes": ["evaluator-hash"],
        "split_manifest_ref": "construct/cells/example/split_manifest.json",
    }

    bert_row = _construct_row_base(**common, cell=bert)
    qwen_row = _construct_row_base(**common, cell=qwen)

    for row, cell in ((bert_row, bert), (qwen_row, qwen)):
        assert (row["model_key"], row["model_id"], row["task"], row["layer"]) == (
            cell.model_key,
            cell.model_id,
            cell.task,
            cell.layer,
        )
        assert construct_row_identity(
            cell,
            edit_kind="dcand_crossfit",
            architecture="linear",
            seed=3,
        ) == {
            "model_key": cell.model_key,
            "model_id": cell.model_id,
            "task": cell.task,
            "layer": cell.layer,
            "edit_kind": "dcand_crossfit",
            "architecture": "linear",
            "candidate_seed": 3,
        }
    assert bert_row != qwen_row


def test_legacy_construct_layout_row_and_wrapper_contract_remain_exact(
    monkeypatch, tmp_path
):
    legacy = _construct_layout(tmp_path, None, legacy=True)
    edit_key = ("dcand_crossfit", "linear", 3)

    assert legacy.artifact_root == tmp_path / "construct"
    assert legacy.checkpoint_root == tmp_path / "checkpoints" / "construct"
    assert legacy.array_root == tmp_path / "arrays" / "construct"
    assert legacy.rows_csv == tmp_path / "construct_check_rows.csv"
    assert legacy.rows_parquet == tmp_path / "construct_check_rows.parquet"
    assert legacy.summary_path == tmp_path / "construct_check_summary.json"
    assert legacy.shard_experiment == "construct_edits"
    assert legacy.shard_key(edit_key) == edit_key
    assert legacy.summary_cell == {
        "model_key": "qwen",
        "task": "sst2",
        "layer": 14,
    }

    row = _construct_row_base(
        context=SimpleNamespace(run_id="test-run"),
        config=SimpleNamespace(config_hash="config-hash"),
        edit_id="dcand_crossfit-linear-seed3",
        edit_kind="dcand_crossfit",
        architecture="linear",
        seed=3,
        evaluation_family="fixed",
        label="target",
        edit_hash="edit-hash",
        candidate_checkpoint_hash="candidate-hash",
        evaluator_checkpoint_hashes=["evaluator-hash"],
        split_manifest_ref="construct/split_manifest.json",
    )
    assert (row["model_key"], row["model_id"], row["task"], row["layer"]) == (
        "qwen",
        "Qwen/Qwen2.5-1.5B",
        "sst2",
        14,
    )

    observed = {}

    def fake_worker(context, config, *, cell, device, legacy_layout):
        observed.update(
            context=context,
            config=config,
            cell=cell,
            device=device,
            legacy_layout=legacy_layout,
        )
        return {"status": "ok"}

    monkeypatch.setattr(runner, "_run_construct_cell", fake_worker)
    context = object()
    config = object()
    device = torch.device("cpu")

    assert run_construct_check(context, config, device=device) == {"status": "ok"}
    assert observed == {
        "context": context,
        "config": config,
        "cell": ConstructCell("qwen", "Qwen/Qwen2.5-1.5B", "sst2", 14),
        "device": device,
        "legacy_layout": True,
    }


class _ToyConstructConfig:
    config_hash = "c" * 64
    tasks = ("sva", "sst2")
    pair_seeds: tuple[int, ...] = ()
    raw: ClassVar[dict[str, object]] = {
        "reproducibility": {
            "master_seed": 20260830,
            "bootstrap_seed": 20260830,
            "bootstrap_draws": 100,
            "permutation_seed": 20260831,
        }
    }
    _models: ClassVar[dict[str, dict[str, object]]] = {
        "bert": {
            "key": "bert",
            "hf_id": "google-bert/bert-base-uncased",
            "middle_layer": 6,
        },
        "qwen": {
            "key": "qwen",
            "hf_id": "Qwen/Qwen2.5-1.5B",
            "middle_layer": 14,
        },
    }
    _edits = (
        ConstructEditKey("dcand_crossfit", "linear", 0),
        ConstructEditKey("dcand_crossfit", "linear", 1),
    )

    def model(self, model_key: str):
        return self._models[model_key]

    def construct_edit_keys(self):
        return self._edits


def _toy_reconstruction(task: str):
    folds = {
        subset: [object() for _ in range(size)]
        for subset, size in (
            ("candidate", 4),
            ("evaluator", 4),
            ("intervention", 8),
            ("test", 2),
        )
    }
    manifest = []
    for subset, examples in folds.items():
        for index, _ in enumerate(examples):
            manifest.append(
                {
                    "subset": subset,
                    "position_in_subset": index,
                    "group_id": f"{task}-{subset}-group-{index}",
                    "example_id": f"{task}-{subset}-example-{index}",
                }
            )
    return SimpleNamespace(
        task_name=task,
        folds=folds,
        manifest=manifest,
        all_data_hash=f"all-data-{task}",
        fold_hashes={subset: f"fold-{task}-{subset}" for subset in folds},
    )


def _toy_subdivision(task: str):
    subsets = (
        "direction_fit",
        "fresh_decoder_fit",
        "orientation_calibration",
        "final_test",
    )
    manifest = [
        {
            "subset": subset,
            "position_in_subset": position,
            "group_id": f"{task}-{subset}-group-{position}",
            "example_id": f"{task}-{subset}-example-{position}",
        }
        for subset in subsets
        for position in range(2)
    ]
    return SimpleNamespace(
        manifest=manifest,
        subset_hashes={subset: f"hash-{task}-{subset}" for subset in subsets},
        diagnostics={"task": task},
    )


def _install_toy_construct_environment(monkeypatch, *, interrupt_once: bool):
    config = _ToyConstructConfig()
    reconstructions = {task: _toy_reconstruction(task) for task in config.tasks}
    state = SimpleNamespace(
        cache_calls=[],
        subdivision_calls=[],
        candidate_calls=[],
        checkpoint_calls=[],
        evaluator_calls=[],
        interrupt_pending=interrupt_once,
    )
    pins = {
        (model["hf_id"], task, model["middle_layer"], tag): (
            f"pin-{model_key}-{task}-{tag}"
        )
        for model_key, model in config._models.items()
        for task in config.tasks
        for tag in ("cand", "eval", "inter")
    }

    monkeypatch.setattr(runner, "_require_ok_report", lambda *_args: {})
    monkeypatch.setattr(
        runner,
        "_reconstruct_tasks",
        lambda _config: reconstructions,
    )
    monkeypatch.setattr(runner, "_preflight_cache_pins", lambda _context: pins)

    def fake_load_cache(
        *,
        model_id,
        task,
        layer,
        tag,
        reconstruction,
        expected_cache_sha256,
    ):
        state.cache_calls.append((model_id, task, layer, tag, reconstruction.task_name))
        assert expected_cache_sha256 == pins[(model_id, task, layer, tag)]
        size = 8 if tag == "inter" else 4
        values = torch.arange(size * 2, dtype=torch.float32).reshape(size, 2)
        return SimpleNamespace(
            X=values + (0.1 if tag == "eval" else 0.0),
            zc=torch.tensor([index % 2 for index in range(size)]),
            ze=torch.tensor([(index + 1) % 2 for index in range(size)]),
            selection=SimpleNamespace(
                cache_sha256=expected_cache_sha256,
                data_hash=f"data-{task}-{tag}",
            ),
        )

    monkeypatch.setattr(runner, "_load_cache", fake_load_cache)

    def fake_subdivision(_examples, *, seed, task_name):
        assert seed == config.raw["reproducibility"]["master_seed"]
        state.subdivision_calls.append(task_name)
        return _toy_subdivision(task_name)

    monkeypatch.setattr(
        runner,
        "subdivide_phase2_intervention",
        fake_subdivision,
    )
    monkeypatch.setattr(
        runner,
        "_subdivision_indices",
        lambda _source, _subdivision: {
            "direction_fit": torch.tensor([0, 1]),
            "fresh_decoder_fit": torch.tensor([2, 3]),
            "orientation_calibration": torch.tensor([4, 5]),
            "final_test": torch.tensor([6, 7]),
        },
    )

    selection = HyperparameterSelection(
        selected_spec=DecoderSpec(architecture="linear"),
        records=[],
        train_indices=torch.tensor([0]),
        validation_indices=torch.tensor([1]),
    )
    monkeypatch.setattr(
        runner,
        "select_linear_hyperparameters",
        lambda **_kwargs: selection,
    )

    def fake_fresh_checkpoint(path: Path, *, metadata, **_kwargs):
        path = Path(path)
        state.checkpoint_calls.append((path, dict(metadata)))
        accuracy = 0.8 if "unedited" in path.parts else 0.7
        checkpoint_hash = sha256_json({"path": path.as_posix(), "metadata": metadata})
        runner.atomic_torch_save(
            path,
            {"checkpoint_sha256": checkpoint_hash, "metadata": dict(metadata)},
        )
        per_example_path = path.with_suffix(".per-example.pt")
        runner.atomic_torch_save(
            per_example_path,
            {"checkpoint_sha256": checkpoint_hash, "metadata": dict(metadata)},
        )
        return (
            {
                "accuracy": accuracy,
                "balanced_accuracy": accuracy,
                "auc": accuracy,
                "log_loss": 1.0 - accuracy,
            },
            checkpoint_hash,
            [{"epoch": 0, "validation_loss": 1.0 - accuracy}],
            0,
            per_example_path,
        )

    monkeypatch.setattr(runner, "_fresh_checkpoint", fake_fresh_checkpoint)
    monkeypatch.setattr(runner, "_construct_attackers", lambda *_args, **_kwargs: [])

    def fixed_probe():
        probe = torch.nn.Linear(2, 2)
        with torch.no_grad():
            probe.weight.zero_()
            probe.bias.zero_()
        return probe

    def fake_evaluators(
        context,
        *,
        checkpoint_root,
        cell_metadata,
        **_kwargs,
    ):
        del context
        state.evaluator_calls.append(
            (
                Path(checkpoint_root),
                None if cell_metadata is None else dict(cell_metadata),
            )
        )
        runner.atomic_torch_save(
            Path(checkpoint_root) / "evaluators.pt",
            {"cell": cell_metadata},
        )
        return [
            SimpleNamespace(
                probes=SimpleNamespace(
                    zc_probe=fixed_probe(),
                    ze_probe=fixed_probe(),
                )
            )
        ]

    monkeypatch.setattr(runner, "_construct_evaluators", fake_evaluators)

    def fake_candidate(
        context,
        *,
        architecture,
        seed,
        checkpoint_root,
        cell_metadata,
        **_kwargs,
    ):
        del context
        identity = (
            "legacy" if cell_metadata is None else str(cell_metadata["model_key"])
        )
        state.candidate_calls.append((identity, int(seed)))
        if identity == "bert" and seed == 1 and state.interrupt_pending:
            state.interrupt_pending = False
            raise KeyboardInterrupt("simulated interruption")
        runner.atomic_torch_save(
            Path(checkpoint_root) / "candidates" / f"{architecture}-seed{seed}.pt",
            {"cell": cell_metadata, "seed": int(seed)},
        )
        return fixed_probe(), sha256_json({"cell": identity, "seed": int(seed)})

    monkeypatch.setattr(runner, "_construct_candidate", fake_candidate)
    monkeypatch.setattr(
        runner,
        "candidate_rank_one_direction",
        lambda _candidate, _X, *, device: (
            torch.tensor([1.0, 0.0], device=device),
            {"direction_norm": 1.0},
        ),
    )

    logits = [[0.0, 1.0], [1.0, 0.0]]
    logits_hash = sha256_tensor(torch.tensor(logits, dtype=torch.float32))
    monkeypatch.setattr(
        runner,
        "_validate_rank_one_reconstruction",
        lambda **_kwargs: {
            "saved_final_logits_sha256": logits_hash,
            "reconstructed_final_logits_sha256": logits_hash,
            "maximum_absolute_logit_difference": 0.0,
            "maximum_absolute_representation_difference": 0.0,
            "tolerance": 1.0e-6,
        },
    )

    def fake_fixed_evaluation(**_kwargs):
        return {
            "final_pre_raw": {"accuracy": 0.8},
            "final_post_raw": {
                "accuracy": 0.7,
                "balanced_accuracy": 0.7,
                "complement_sensitivity_accuracy": 0.7,
                "auc": 0.7,
                "log_loss": 0.3,
            },
            "final_post_oriented": {
                "accuracy": 0.7,
                "auc": 0.7,
                "log_loss": 0.3,
            },
            "C_raw": 0.4,
            "C_orientation": 0.4,
            "orientation": {"sign": 1},
            "per_example": {"final_post": {"logits": logits}},
        }

    monkeypatch.setattr(
        runner,
        "evaluate_fixed_evaluator_edit",
        fake_fixed_evaluation,
    )
    return config, state


def _new_context(tmp_path: Path, config: _ToyConstructConfig) -> RunContext:
    return RunContext.create(
        output_root=tmp_path,
        config_hash=config.config_hash,
        git_commit="d" * 40,
        timestamp=datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc),
    )


def test_two_construct_cells_interrupt_resume_without_cross_cell_collisions(
    monkeypatch, tmp_path
):
    config, state = _install_toy_construct_environment(
        monkeypatch,
        interrupt_once=True,
    )
    bert, qwen = _cells()
    context = _new_context(tmp_path, config)
    try:
        with pytest.raises(KeyboardInterrupt, match="simulated interruption"):
            run_construct_cell(
                context,
                config,
                cell=bert,
                device=torch.device("cpu"),
            )

        bert_layout = _construct_layout(context.run_dir, bert)
        qwen_layout = _construct_layout(context.run_dir, qwen)
        first_key = bert_layout.shard_key(config._edits[0])
        second_key = bert_layout.shard_key(config._edits[1])
        assert context.completed_keys(
            bert_layout.shard_experiment,
            expected_keys=(first_key, second_key),
        ) == {first_key}

        bert_summary = run_construct_cell(
            context,
            config,
            cell=bert,
            device=torch.device("cpu"),
        )
        candidate_calls_after_resume = list(state.candidate_calls)
        assert (
            run_construct_cell(
                context,
                config,
                cell=bert,
                device=torch.device("cpu"),
            )
            == bert_summary
        )
        assert state.candidate_calls == candidate_calls_after_resume

        qwen_summary = run_construct_cell(
            context,
            config,
            cell=qwen,
            device=torch.device("cpu"),
        )

        assert state.candidate_calls.count(("bert", 0)) == 1
        assert state.candidate_calls.count(("bert", 1)) == 2
        assert state.candidate_calls.count(("qwen", 0)) == 1
        assert state.candidate_calls.count(("qwen", 1)) == 1
        assert context.completed_keys(
            bert_layout.shard_experiment,
            expected_keys=(first_key, second_key),
        ) == {first_key, second_key}
        assert context.completed_keys(
            qwen_layout.shard_experiment,
            expected_keys=tuple(qwen_layout.shard_key(edit) for edit in config._edits),
        ) == {qwen_layout.shard_key(edit) for edit in config._edits}
        assert bert_summary["cell"] == {
            "model_key": bert.model_key,
            "model_id": bert.model_id,
            "task": bert.task,
            "layer": bert.layer,
        }
        assert qwen_summary["cell"] == {
            "model_key": qwen.model_key,
            "model_id": qwen.model_id,
            "task": qwen.task,
            "layer": qwen.layer,
        }

        bert_rows = pd.read_parquet(bert_layout.rows_parquet)
        qwen_rows = pd.read_parquet(qwen_layout.rows_parquet)
        assert set(
            bert_rows[["model_key", "model_id", "task", "layer"]].itertuples(
                index=False, name=None
            )
        ) == {(bert.model_key, bert.model_id, bert.task, bert.layer)}
        assert set(
            qwen_rows[["model_key", "model_id", "task", "layer"]].itertuples(
                index=False, name=None
            )
        ) == {(qwen.model_key, qwen.model_id, qwen.task, qwen.layer)}
        assert {(call[0], call[1], call[2], call[4]) for call in state.cache_calls} >= {
            (bert.model_id, bert.task, bert.layer, bert.task),
            (qwen.model_id, qwen.task, qwen.layer, qwen.task),
        }
        assert {bert.task, qwen.task} <= set(state.subdivision_calls)
        assert (
            json.loads(
                (bert_layout.artifact_root / "split_manifest.json").read_text(
                    encoding="utf-8"
                )
            )["source_data_hash"]
            == f"all-data-{bert.task}"
        )
        assert (
            json.loads(
                (qwen_layout.artifact_root / "split_manifest.json").read_text(
                    encoding="utf-8"
                )
            )["source_data_hash"]
            == f"all-data-{qwen.task}"
        )
    finally:
        context.close()


def test_legacy_construct_run_preserves_golden_paths_keys_and_payloads(
    monkeypatch, tmp_path
):
    config, state = _install_toy_construct_environment(
        monkeypatch,
        interrupt_once=False,
    )
    context = _new_context(tmp_path, config)
    try:
        summary = run_construct_check(
            context,
            config,
            device=torch.device("cpu"),
        )
        layout = _construct_layout(context.run_dir, None, legacy=True)

        assert layout.rows_csv.is_file()
        assert layout.rows_parquet.is_file()
        assert layout.summary_path.is_file()
        assert (layout.artifact_root / "standardization.pt").is_file()
        assert (layout.artifact_root / "split_manifest.json").is_file()
        assert (layout.artifact_root / "hyperparameter_selection.json").is_file()
        assert (layout.artifact_root / "fresh_decoder_baselines.json").is_file()
        assert (layout.checkpoint_root / "evaluators.pt").is_file()
        assert (layout.checkpoint_root / "candidates" / "linear-seed0.pt").is_file()
        assert (
            layout.array_root / "dcand_crossfit-linear-seed0" / "evaluator0-target.json"
        ).is_file()
        assert context.completed_keys(
            "construct_edits",
            expected_keys=config._edits,
        ) == {tuple(edit) for edit in config._edits}

        first_payload = context.load_json_shard(
            "construct_edits",
            config._edits[0],
        )
        assert "cell" not in first_payload
        assert first_payload["rows"]
        assert summary["cell"] == {
            "model_key": "qwen",
            "task": "sst2",
            "layer": 14,
        }
        assert "model_id" not in summary["cell"]
        for filename in (
            "split_manifest.json",
            "hyperparameter_selection.json",
            "fresh_decoder_baselines.json",
        ):
            payload = json.loads(
                (layout.artifact_root / filename).read_text(encoding="utf-8")
            )
            assert "cell" not in payload
        fixed_array = json.loads(
            (
                layout.array_root
                / "dcand_crossfit-linear-seed0"
                / "evaluator0-target.json"
            ).read_text(encoding="utf-8")
        )
        assert "cell" not in fixed_array
        direction = torch.load(
            layout.artifact_root / "directions" / "dcand_crossfit-linear-seed0.pt",
            map_location="cpu",
            weights_only=True,
        )
        assert "cell" not in direction
        assert state.evaluator_calls
        assert state.evaluator_calls[0] == (layout.checkpoint_root, None)
        assert all(metadata is None for _, metadata in state.evaluator_calls)
        assert state.cache_calls[:3] == [
            ("Qwen/Qwen2.5-1.5B", "sst2", 14, tag, "sst2")
            for tag in ("cand", "eval", "inter")
        ]
    finally:
        context.close()
