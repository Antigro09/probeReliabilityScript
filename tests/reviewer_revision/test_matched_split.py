from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.reviewer_revision.experiments import (
    archived_baseline_summary,
    score_conditions_for_edit,
    validate_archived_baseline,
    validate_retrained_baseline,
    validate_unique_rows,
)

ARCHIVE_PATH = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "reviewer_revision"
    / "attacker_evaluator_reference.json"
)


class _AxisProbe(torch.nn.Module):
    def __init__(self, axis: int, sign: float = 1.0):
        super().__init__()
        weight = torch.zeros(2, 3)
        weight[0, axis] = -sign
        weight[1, axis] = sign
        self.register_buffer("weight", weight)

    def forward(self, x):
        return x @ self.weight.T


def test_one_edit_tensor_is_shared_by_matched_and_split_rows():
    X_pre = torch.tensor(
        [[-2.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [2.0, 1.0, 0.0]]
    )
    target = torch.tensor([0, 0, 1, 1])
    control = torch.tensor([0, 1, 0, 1])
    X_post = X_pre.clone()
    X_post[:, 0] *= -1

    rows = score_conditions_for_edit(
        X_pre=X_pre,
        X_post=X_post,
        target_labels=target,
        control_labels=control,
        matched_target_probe=_AxisProbe(0),
        matched_control_probe=_AxisProbe(1),
        split_target_probe=_AxisProbe(0),
        split_control_probe=_AxisProbe(1),
        device=torch.device("cpu"),
        common={"model_key": "tiny", "task": "toy", "layer": 1, "pair_seed": 0, "method": "alterrep"},
    )

    assert [row["condition"] for row in rows] == ["matched", "split"]
    assert rows[0]["edit_hash"] == rows[1]["edit_hash"]
    assert rows[0]["target_acc_pre"] == 1.0
    assert rows[0]["target_acc_post"] == 0.0
    assert rows[0]["C"] == 1.0
    assert rows[0]["S"] == 1.0
    validate_unique_rows(rows)


def test_duplicate_scientific_row_key_is_rejected():
    row = {
        "model_key": "m",
        "task": "sva",
        "layer": 1,
        "pair_seed": 0,
        "method": "alterrep",
        "condition": "matched",
    }
    with pytest.raises(ValueError, match="duplicate"):
        validate_unique_rows([row, dict(row)])


def test_archived_fixture_reproduces_locked_aggregate():
    summary = archived_baseline_summary(ARCHIVE_PATH)
    assert summary["n_cells"] == 12
    assert summary["matched"] == pytest.approx(0.9829058376336399, abs=1e-15)
    assert summary["split"] == pytest.approx(0.639718191210888, abs=1e-15)
    assert summary["gap"] == pytest.approx(0.34318764642275196, abs=1e-15)
    assert summary["positive_gap_cells"] == 7
    assert summary["zero_gap_cells"] == 5
    assert summary["fgsm_pgd_ceiling_reproduced"] is True
    assert summary["fgsm_pgd_ceiling_pair_values"] == 240


def test_archived_gate_recomputes_pair_arrays_and_rejects_stale_stored_means(tmp_path):
    payload = json.loads(ARCHIVE_PATH.read_text(encoding="utf-8"))
    first_cell = next(iter(payload["cells"].values()))
    first_cell["methods"]["AlterRep"]["matched_C_mean"] = 0.123
    path = tmp_path / "stale-mean.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="stored mean"):
        validate_archived_baseline(path)


def test_archived_gate_rejects_missing_attack_ceiling_pair(tmp_path):
    payload = json.loads(ARCHIVE_PATH.read_text(encoding="utf-8"))
    first_cell = next(iter(payload["cells"].values()))
    first_cell["methods"]["PGD"]["split_C_pairs"][0] = 0.99
    first_cell["methods"]["PGD"]["split_C_mean"] = 0.998
    path = tmp_path / "broken-ceiling.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="ceiling"):
        validate_archived_baseline(path)


def test_retrained_baseline_gate_compares_current_pair_outputs(tmp_path):
    current = json.loads(ARCHIVE_PATH.read_text(encoding="utf-8"))

    report = validate_retrained_baseline(ARCHIVE_PATH, current)

    assert report["status"] == "ok"
    assert report["current"]["fgsm_pgd_ceiling_pair_values"] == 240
    assert max(report["aggregate_absolute_deviations"].values()) == 0.0

    first = next(iter(current["cells"].values()))
    first["methods"]["AlterRep"]["split_C_pairs"] = [0.0] * 5
    first["methods"]["AlterRep"]["split_C_mean"] = 0.0
    first["methods"]["AlterRep"]["gap"] = first["methods"]["AlterRep"][
        "matched_C_mean"
    ]
    with pytest.raises(ValueError, match="aggregate tolerance|sign pattern"):
        validate_retrained_baseline(ARCHIVE_PATH, current)
