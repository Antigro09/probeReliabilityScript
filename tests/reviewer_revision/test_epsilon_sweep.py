from __future__ import annotations

import pandas as pd
import pytest
import torch

from src.probes import LinearProbe
from src.reviewer_revision.artifacts import RunContext
from src.reviewer_revision.experiments import (
    AttackerEvaluatorPair,
    ProbePair,
    attack_edit,
    pgd_step_size,
    realized_linf_norm,
    validate_linf_edit,
)
from src.reviewer_revision.runner import (
    _augment_epsilon_metrics,
    _summarize_epsilon_half_pattern,
    _validate_epsilon_baseline_against_matched,
)


@pytest.mark.parametrize("method", ["fgsm", "pgd"])
def test_epsilon_zero_is_exact_noop(method):
    generator = torch.Generator().manual_seed(31)
    X = torch.randn(32, 5, generator=generator)
    y = (X[:, 0] > 0).long()
    probe = LinearProbe(5)

    edited = attack_edit(
        method, X, y, probe, device=torch.device("cpu"), epsilon=0.0,
        pgd_steps=10,
    )

    assert torch.equal(edited, X)
    assert realized_linf_norm(X, edited) == 0.0


@pytest.mark.parametrize("method", ["fgsm", "pgd"])
def test_attack_respects_linf_ball(method):
    generator = torch.Generator().manual_seed(32)
    X = torch.randn(32, 5, generator=generator)
    y = (X[:, 0] > 0).long()
    probe = LinearProbe(5)
    epsilon = 0.03125

    edited = attack_edit(
        method, X, y, probe, device=torch.device("cpu"), epsilon=epsilon,
        pgd_steps=10,
    )

    assert validate_linf_edit(X, edited, epsilon) <= epsilon + 1e-6
    assert torch.isfinite(edited).all()


def test_locked_pgd_step_size_rule():
    assert pgd_step_size(0.0) == 0.0
    assert pgd_step_size(0.5) == pytest.approx(0.1)
    assert pgd_step_size(0.03125) == pytest.approx(0.00625)


def test_epsilon_artifact_preserves_shared_edit_and_per_example_provenance(tmp_path):
    target_probe = LinearProbe(3)
    control_probe = LinearProbe(3)
    attacker = ProbePair(target_probe, control_probe, "a" * 64, "b" * 64, 0)
    evaluator = ProbePair(LinearProbe(3), LinearProbe(3), "c" * 64, "d" * 64, 1)
    pair = AttackerEvaluatorPair(attacker, evaluator, pair_seed=0)
    X_pre = torch.tensor([[-1.0, 0.0, 0.5], [1.0, 0.5, -0.5]])
    X_post = X_pre + torch.tensor([[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]])
    target = torch.tensor([0, 1])
    control = torch.tensor([1, 0])
    rows = [
        {
            "model_key": "tiny",
            "task": "toy",
            "layer": 1,
            "pair_seed": 0,
            "condition": condition,
            "target_acc_pre": 1.0,
            "target_acc_post": 0.5,
            "control_acc_pre": 1.0,
            "control_acc_post": 1.0,
            "C": 1.0,
        }
        for condition in ("matched", "split")
    ]

    with RunContext.create(
        output_root=tmp_path,
        config_hash="e" * 64,
        git_commit="f" * 40,
    ) as context:
        _augment_epsilon_metrics(
            context,
            rows,
            pair=pair,
            X_pre=X_pre,
            X_post=X_post,
            target_labels=target,
            control_labels=control,
            example_ids=["ex-0", "ex-1"],
            split_name="score",
            source_cache_sha256="1" * 64,
            source_data_hash="2" * 16,
            source_indices=[0, 1],
            epsilon=0.01,
            method="fgsm",
            device=torch.device("cpu"),
        )

        assert rows[0]["edit_artifact_ref"] == rows[1]["edit_artifact_ref"]
        edit_payload = torch.load(
            context.run_dir / rows[0]["edit_artifact_ref"], weights_only=True
        )
        assert edit_payload["example_ids"] == ["ex-0", "ex-1"]
        assert edit_payload["split_names"] == ["score", "score"]
        assert torch.equal(edit_payload["representation_delta"], X_post - X_pre)
        assert torch.equal(X_pre + edit_payload["representation_delta"], X_post)
        assert "edited_representations" not in edit_payload
        assert edit_payload["source_cache_sha256"] == "1" * 64

        for row in rows:
            evaluation_ref = row["per_example_artifact_ref"]
            evaluation_payload = torch.load(
                context.run_dir / evaluation_ref, weights_only=True
            )
            assert evaluation_payload["example_ids"] == ["ex-0", "ex-1"]
            assert "target_probabilities_post" in evaluation_payload
            assert "control_probabilities_post" in evaluation_payload


def test_epsilon_half_gate_requires_same_edit_hash_and_metrics_as_experiment_a():
    common = {
        "model_key": "qwen",
        "task": "sst2",
        "layer": 14,
        "pair_seed": 0,
        "method": "fgsm",
        "condition": "matched",
        "edit_hash": "a" * 64,
        "status": "ok",
        "C": 0.8,
        "S": 0.8,
        "H": 0.888,
        "target_acc_pre": 0.9,
        "target_acc_post": 0.5,
        "control_acc_pre": 0.8,
        "control_acc_post": 0.75,
    }
    matched = pd.DataFrame([common, {**common, "layer": 1}])
    epsilon = pd.DataFrame(
        [{**common, "epsilon": 0.5, "epsilon_scope": "required_middle"}]
    )
    report = _validate_epsilon_baseline_against_matched(
        matched, epsilon, expected_rows=1
    )
    assert report["passed"] is True
    assert report["rows"] == 1
    pattern = _summarize_epsilon_half_pattern(epsilon, expected_rows=1)
    assert pattern["ceiling_rows"] == 0
    assert pattern["ceiling_fraction_among_defined"] == 0.0

    epsilon.loc[0, "edit_hash"] = "b" * 64
    with pytest.raises(ValueError, match="edit hash"):
        _validate_epsilon_baseline_against_matched(matched, epsilon, expected_rows=1)
