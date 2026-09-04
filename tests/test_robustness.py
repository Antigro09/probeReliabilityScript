"""Unit tests for schema-v3 metrics, edits, cross-fitting, and group splits."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from src.probes import LinearProbe
from src.robustness import (
    SubspaceEdit,
    binary_scores_from_probabilities,
    cross_fitted_candidate_edit,
    distortion_matched_random_edit,
    fit_leace_binary,
    grouped_stratified_split,
    normalize_gender_minimal_pair,
    orientation_robust_cs,
    position_variant,
)
from src.tasks import Example
from src.ws5_repaired import candidate_direction_details

DEVICE = torch.device("cpu")


def _linear_probe(direction):
    probe = LinearProbe(direction.numel())
    with torch.no_grad():
        probe.linear.weight.zero_()
        probe.linear.bias.zero_()
        probe.linear.weight[1] = direction
    return probe


def test_inverted_predictions_remain_decodable():
    y = torch.tensor([0, 0, 1, 1])
    inverted = torch.tensor([0.99, 0.90, 0.10, 0.01])
    scores = binary_scores_from_probabilities(inverted, y)
    assert scores.accuracy == pytest.approx(0.0)
    assert scores.complemented_accuracy == pytest.approx(1.0)
    assert scores.roc_auc == pytest.approx(0.0)
    assert scores.oriented_roc_auc == pytest.approx(1.0)


def test_orientation_robust_completeness_does_not_reward_inversion():
    y = torch.tensor([0, 0, 1, 1])
    clean = binary_scores_from_probabilities(torch.tensor([0.01, 0.10, 0.90, 0.99]), y)
    inverted = binary_scores_from_probabilities(torch.tensor([0.99, 0.90, 0.10, 0.01]), y)
    chance = binary_scores_from_probabilities(torch.tensor([0.4, 0.6, 0.4, 0.6]), y)

    preserved = orientation_robust_cs(clean, inverted, clean, clean)
    removed = orientation_robust_cs(clean, chance, clean, clean)
    assert preserved["C_auc_oriented"] == pytest.approx(0.0)
    assert preserved["R_auc_oriented"] == pytest.approx(0.0)
    assert removed["C_auc_oriented"] == pytest.approx(1.0)
    assert removed["S_auc_oriented"] == pytest.approx(1.0)
    assert removed["R_auc_oriented"] == pytest.approx(1.0)


def test_binary_leace_removes_empirical_target_cross_covariance():
    generator = torch.Generator().manual_seed(3)
    y = torch.randint(0, 2, (300,), generator=generator)
    X = torch.randn(300, 12, generator=generator)
    X[:, 0] += (2 * y - 1).float() * 2.0
    X[:, 3] += (2 * y - 1).float() * 0.5
    edit = fit_leace_binary(X, y)
    edited = edit.apply(X)
    centered_y = y.float() - y.float().mean()
    cross_cov = (edited - edited.mean(0)).T @ centered_y / len(y)
    assert float(cross_cov.norm()) < 1e-4


def test_distortion_matched_random_control_matches_frobenius_norm():
    generator = torch.Generator().manual_seed(4)
    X = torch.randn(100, 16, generator=generator)
    Q = torch.linalg.qr(torch.randn(16, 2, generator=generator), mode="reduced")[0]
    target = SubspaceEdit(Q)
    matched, meta = distortion_matched_random_edit(target, X, seed=11)
    assert meta["matched_distortion"] == pytest.approx(meta["target_distortion"], rel=1e-5)
    assert matched.rank == target.rank


def test_linear_candidate_reports_effective_rank_one_when_rank_five_requested():
    generator = torch.Generator().manual_seed(5)
    direction = torch.randn(10, generator=generator)
    probe = _linear_probe(direction)
    detail = candidate_direction_details(probe, torch.randn(80, 10, generator=generator), DEVICE, r=5)
    assert detail["requested_rank"] == 5
    assert detail["effective_rank"] == 1
    assert detail["Q"].shape == (10, 1)


def test_cross_fitted_edit_never_uses_heldout_examples_for_direction():
    direction = torch.zeros(8)
    direction[0] = 1.0
    probe = _linear_probe(direction)
    X = torch.randn(60, 8, generator=torch.Generator().manual_seed(6))
    result = cross_fitted_candidate_edit(probe, X, DEVICE, rank=1, folds=5, seed=17)
    assert not result["degenerate"]
    assert result["X_post"].shape == X.shape
    assert float(result["X_post"][:, 0].abs().max()) < 1e-4
    assert all(record["n_fit"] + record["n_score"] == len(X) for record in result["folds"])


def test_gender_minimal_pairs_cannot_cross_grouped_folds():
    examples = []
    for i in range(40):
        ze = i % 2
        examples.extend([
            Example(f"The doctor {i} said he was ready.", 0, ze),
            Example(f"The doctor {i} said she was ready.", 1, ze),
        ])
    fractions = {
        "candidate": 0.30, "evaluator": 0.20, "direction": 0.15,
        "decoder": 0.20, "test": 0.15,
    }
    folds, diagnostics = grouped_stratified_split(
        examples, fractions, seed=42, task_name="gender", group_mode="minimal_pair"
    )
    owner = {}
    for fold, rows in folds.items():
        for row in rows:
            pair = normalize_gender_minimal_pair(row.sentence)
            assert owner.setdefault(pair, fold) == fold
    assert diagnostics["n_groups"] == 40
    assert sum(diagnostics["fold_sizes"].values()) == 80
    assert all(len(rows) > 0 for rows in folds.values())


def test_pre_pronoun_control_removes_visible_gender_token():
    examples = [Example("The doctor said that she was ready.", 1, 0)]
    variant = position_variant(examples, "gender", "pre-pronoun")
    assert variant[0].sentence == "The doctor said that"
    assert "she" not in variant[0].sentence.lower().split()

