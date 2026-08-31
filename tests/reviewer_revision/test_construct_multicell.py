from __future__ import annotations

from types import SimpleNamespace

import torch

from src.reviewer_revision import runner
from src.reviewer_revision.extension_config import ConstructCell
from src.reviewer_revision.runner import (
    _construct_layout,
    _construct_row_base,
    construct_artifact_root,
    construct_row_identity,
    run_construct_check,
)


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
