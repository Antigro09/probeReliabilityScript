"""
Tests for WS7: TVD-based reliability estimators (arXiv:2408.15510 Eqs 1-4).

These tests pin the exact formula behavior we coded — they are the executable
contract, and (per src/tvd.py's banner) they, not the paper, define what our
numbers mean until the formulas are VERIFIED against arXiv:2408.15510.

Coverage:
    - tvd(p, p) == 0; tvd of opposite point masses == 1; batched TVD.
    - softmax_dist normalizes over the class dim.
    - Eq2 nullifying completeness: C == 1 when post is uniform (binary),
      C == 0 when post is a confident point mass.
    - Eq1 counterfactual completeness: C == 1 when post is one-hot on the
      flipped label; raises for k > 2.
    - Eq3 selectivity == 1 when pre == post distributions.
    - Eq4 reliability harmonic-mean and null propagation.
    - Decodability-floor nulling in compute_tvd_metrics via a tiny LinearProbe
      oracle and a ValidationProbes stub.

The tvd()/softmax_dist()/Eq-kernel tests are torch-only and dependency-light.

Run with pytest, or directly: python -m tests.test_tvd
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch

from src.tvd import (
    tvd,
    softmax_dist,
    nullifying_completeness,
    counterfactual_completeness,
    selectivity_tvd,
    reliability,
    compute_tvd_metrics,
    TVDMetrics,
)
from src.metrics import ValidationProbes, DECODABILITY_FLOOR
from src.probes import LinearProbe

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# tvd() primitive  (torch-only, light)
# ---------------------------------------------------------------------------

def test_tvd_identical_is_zero():
    p = torch.tensor([0.2, 0.3, 0.5])
    assert tvd(p, p) == pytest.approx(0.0)


def test_tvd_opposite_point_masses_is_one():
    p = torch.tensor([1.0, 0.0])
    q = torch.tensor([0.0, 1.0])
    assert tvd(p, q) == pytest.approx(1.0)


def test_tvd_half_overlap():
    # 0.5 * (|0.5-0| + |0.5-1|) = 0.5 * (0.5 + 0.5) = 0.5
    p = torch.tensor([0.5, 0.5])
    q = torch.tensor([0.0, 1.0])
    assert tvd(p, q) == pytest.approx(0.5)


def test_tvd_is_symmetric():
    p = torch.tensor([0.1, 0.6, 0.3])
    q = torch.tensor([0.4, 0.4, 0.2])
    assert tvd(p, q) == pytest.approx(tvd(q, p))


def test_tvd_batched_returns_per_row():
    p = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    q = torch.tensor([[0.0, 1.0], [0.5, 0.5]])
    out = tvd(p, q)
    assert out.shape == (2,)
    assert out[0].item() == pytest.approx(1.0)
    assert out[1].item() == pytest.approx(0.0)


def test_tvd_scalar_return_type_is_float():
    assert isinstance(tvd(torch.tensor([0.5, 0.5]), torch.tensor([0.5, 0.5])), float)


# ---------------------------------------------------------------------------
# softmax_dist()  (torch-only, light)
# ---------------------------------------------------------------------------

def test_softmax_dist_normalizes():
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    probs = softmax_dist(logits)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-6)
    # Uniform logits -> uniform distribution.
    assert torch.allclose(probs[1], torch.full((3,), 1 / 3), atol=1e-6)


def test_softmax_dist_monotonic():
    probs = softmax_dist(torch.tensor([3.0, 1.0]))
    assert probs[0] > probs[1]


# ---------------------------------------------------------------------------
# Eq2 nullifying completeness
# ---------------------------------------------------------------------------

def test_eq2_uniform_post_gives_completeness_one():
    # Binary uniform post -> C = 1 - 2 * delta(U, U) = 1.
    post = torch.full((16, 2), 0.5)
    assert nullifying_completeness(post) == pytest.approx(1.0)


def test_eq2_confident_point_mass_gives_completeness_zero():
    # Binary point mass -> delta(point, U) = 0.5, factor 2 -> C = 0.
    post = torch.zeros(16, 2)
    post[:, 0] = 1.0
    assert nullifying_completeness(post) == pytest.approx(0.0)


def test_eq2_binary_factor_is_two():
    # Halfway distribution (0.75, 0.25): delta to U=(0.5,0.5) is 0.25,
    # C = 1 - 2 * 0.25 = 0.5.
    post = torch.tensor([[0.75, 0.25]] * 8)
    assert nullifying_completeness(post) == pytest.approx(0.5)


def test_eq2_requires_two_classes():
    with pytest.raises(ValueError):
        nullifying_completeness(torch.ones(4, 1))


def test_eq2_per_example_mode_matches_dist_when_homogeneous():
    post = torch.full((10, 2), 0.5)
    assert nullifying_completeness(post, mode="dist") == pytest.approx(
        nullifying_completeness(post, mode="per_example")
    )


# ---------------------------------------------------------------------------
# Eq1 counterfactual completeness
# ---------------------------------------------------------------------------

def test_eq1_perfect_counterfactual_flip_gives_one():
    # True zc all 0; flipped target z' = 1. Post one-hot on class 1 -> C = 1.
    zc = torch.zeros(12, dtype=torch.long)
    post = torch.zeros(12, 2)
    post[:, 1] = 1.0
    assert counterfactual_completeness(post, zc) == pytest.approx(1.0)


def test_eq1_no_flip_gives_zero():
    # Post still one-hot on the TRUE class 0, target is class 1 -> delta = 1.
    zc = torch.zeros(12, dtype=torch.long)
    post = torch.zeros(12, 2)
    post[:, 0] = 1.0
    assert counterfactual_completeness(post, zc) == pytest.approx(0.0)


def test_eq1_raises_for_multiclass():
    zc = torch.zeros(4, dtype=torch.long)
    post = torch.full((4, 3), 1 / 3)
    with pytest.raises(ValueError):
        counterfactual_completeness(post, zc)


def test_eq1_per_example_flip():
    # Mixed labels; each example's post one-hots onto its own flipped target.
    zc = torch.tensor([0, 1, 0, 1])
    post = torch.zeros(4, 2)
    post[torch.arange(4), 1 - zc] = 1.0
    assert counterfactual_completeness(post, zc, mode="per_example") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Eq3 selectivity
# ---------------------------------------------------------------------------

def test_eq3_unchanged_distribution_gives_selectivity_one():
    pre = softmax_dist(torch.randn(20, 2))
    assert selectivity_tvd(pre, pre) == pytest.approx(1.0)


def test_eq3_max_move_gives_selectivity_zero():
    # Clean P is a point mass at class 0; post moves fully to class 1.
    # m = max(1 - 0, 1) = 1; delta = 1 -> S = 0.
    pre = torch.zeros(8, 2)
    pre[:, 0] = 1.0
    post = torch.zeros(8, 2)
    post[:, 1] = 1.0
    assert selectivity_tvd(pre, post) == pytest.approx(0.0)


def test_eq3_shape_mismatch_raises():
    with pytest.raises(ValueError):
        selectivity_tvd(torch.rand(4, 2), torch.rand(5, 2))


def test_eq3_per_example_unchanged_is_one():
    pre = softmax_dist(torch.randn(15, 2))
    assert selectivity_tvd(pre, pre, mode="per_example") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Eq4 reliability
# ---------------------------------------------------------------------------

def test_eq4_harmonic_mean():
    # C = S = 0.5 -> R = 0.5; C=1,S=0.5 -> R = 2/3.
    assert reliability(0.5, 0.5) == pytest.approx(0.5)
    assert reliability(1.0, 0.5) == pytest.approx(2.0 / 3.0)


def test_eq4_zero_sum_is_zero_not_nan():
    assert reliability(0.0, 0.0) == 0.0


def test_eq4_null_propagates():
    assert reliability(None, 0.9) is None
    assert reliability(0.9, None) is None
    assert reliability(None, None) is None


# ---------------------------------------------------------------------------
# compute_tvd_metrics: floor nulling + wiring
#   Uses a tiny real LinearProbe as oracle + a ValidationProbes stub.
# ---------------------------------------------------------------------------

def _perfect_binary_probe(dim: int = 4) -> LinearProbe:
    """A LinearProbe whose logits read off feature 0 with a strong margin, so
    it decodes label = 1[x[0] > 0] essentially perfectly on separable data."""
    probe = LinearProbe(dim)
    with torch.no_grad():
        probe.linear.weight.zero_()
        probe.linear.bias.zero_()
        # class-1 logit tracks +x[0], class-0 logit tracks -x[0], big gain.
        probe.linear.weight[1, 0] = 10.0
        probe.linear.weight[0, 0] = -10.0
    probe.eval()
    return probe


def _chance_probe(dim: int = 4) -> LinearProbe:
    """A LinearProbe with zero weights: always emits uniform logits, so its
    accuracy sits at chance and it triggers the decodability floor."""
    probe = LinearProbe(dim)
    with torch.no_grad():
        probe.linear.weight.zero_()
        probe.linear.bias.zero_()
    probe.eval()
    return probe


def _separable_data(n: int = 64, dim: int = 4, seed: int = 0, margin: float = 2.0):
    """Two well-separated Gaussian blobs along feature 0 with a class margin,
    so the perfect oracle is genuinely CONFIDENT (softmax near a point mass),
    not just correct-by-argmax.

    NOTE: labels are BALANCED and the oracle is confident, so the class-0 and
    class-1 point masses average to the uniform marginal. That is exactly the
    case where the mean-of-distributions aggregation (mode='dist') is
    degenerate for nullifying completeness (it reports C~1 for a no-op),
    while the DEFAULT per-example aggregation correctly reports C~0. See
    test_compute_dist_mode_completeness_is_degenerate_on_balanced_data below.
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, dim, generator=g)
    labels = (torch.arange(n) % 2).long()          # balanced 0/1
    # Push feature 0 to +margin for class 1, -margin for class 0.
    X[:, 0] = X[:, 0].abs() * torch.where(labels == 1, 1.0, -1.0) + \
        margin * torch.where(labels == 1, 1.0, -1.0)
    return X, labels


def test_compute_floor_nulls_completeness_when_zc_not_decodable():
    """Zc oracle at chance on clean reps -> completeness (and R) null."""
    X, labels = _separable_data()
    zc_probe = _chance_probe(X.shape[1])   # can't decode -> below floor
    ze_probe = _perfect_binary_probe(X.shape[1])
    vp = ValidationProbes(zc_probe=zc_probe, ze_probe=ze_probe,
                          acc_zc=0.5, acc_ze=1.0)

    m = compute_tvd_metrics(vp, X_pre=X, X_post=X, zc=labels, ze=labels,
                            device=DEVICE)
    assert m.completeness_null is None
    assert m.completeness_cf is None
    assert m.reliability_null is None
    assert m.reliability_cf is None
    assert m.acc_zc_pre < DECODABILITY_FLOOR


def test_compute_floor_nulls_selectivity_when_ze_not_decodable():
    """Ze oracle at chance on clean reps -> selectivity (and R) null."""
    X, labels = _separable_data()
    zc_probe = _perfect_binary_probe(X.shape[1])
    ze_probe = _chance_probe(X.shape[1])   # can't decode -> below floor
    vp = ValidationProbes(zc_probe=zc_probe, ze_probe=ze_probe,
                          acc_zc=1.0, acc_ze=0.5)

    m = compute_tvd_metrics(vp, X_pre=X, X_post=X, zc=labels, ze=labels,
                            device=DEVICE)
    assert m.selectivity is None
    assert m.reliability_null is None
    assert m.completeness_null is not None  # Zc side healthy
    assert m.acc_ze_pre < DECODABILITY_FLOOR


def test_compute_no_op_intervention_has_perfect_selectivity():
    """X_post == X_pre: the non-target Ze distribution is unchanged -> S = 1."""
    X, labels = _separable_data()
    zc_probe = _perfect_binary_probe(X.shape[1])
    ze_probe = _perfect_binary_probe(X.shape[1])
    vp = ValidationProbes(zc_probe=zc_probe, ze_probe=ze_probe,
                          acc_zc=1.0, acc_ze=1.0)

    m = compute_tvd_metrics(vp, X_pre=X, X_post=X, zc=labels, ze=labels,
                            device=DEVICE)
    assert m.selectivity == pytest.approx(1.0)
    # A no-op does NOT nullify Zc. Under the DEFAULT per-example aggregation,
    # each confident example's distribution is far from uniform, so the
    # nullifying completeness is ~0 (correct), not spuriously high.
    assert m.completeness_null is not None
    assert m.completeness_null == pytest.approx(0.0, abs=1e-3)


def test_compute_dist_mode_completeness_is_degenerate_on_balanced_data():
    """Regression pin for the mode='dist' marginal-cancellation degeneracy.

    On balanced, confident, PERFECTLY-DECODABLE clean reps a NO-OP
    intervention should have completeness ~0. The default per-example mode
    gets this right; mode='dist' is DEGENERATE — the balanced point masses
    average to the uniform marginal, so Eq2 reports a spurious C ~ 1. This
    test documents (does not endorse) that behavior so nobody flips the
    default to 'dist' without noticing. See src/tvd.py choice (a).
    """
    X, labels = _separable_data()
    probe = _perfect_binary_probe(X.shape[1])
    vp = ValidationProbes(zc_probe=probe, ze_probe=probe,
                          acc_zc=1.0, acc_ze=1.0)

    m_default = compute_tvd_metrics(vp, X_pre=X, X_post=X, zc=labels, ze=labels,
                                    device=DEVICE)
    m_dist = compute_tvd_metrics(vp, X_pre=X, X_post=X, zc=labels, ze=labels,
                                 device=DEVICE, mode="dist")

    # Correct (default): no-op -> completeness ~ 0.
    assert m_default.completeness_null == pytest.approx(0.0, abs=1e-3)
    # Degenerate (dist): no-op -> spurious completeness ~ 1.
    assert m_dist.completeness_null == pytest.approx(1.0, abs=1e-3)


def test_compute_zeroed_reps_drive_completeness_high():
    """Zeroing the reps sends the linear oracle (zero bias) to uniform logits,
    so the Zc oracle distribution collapses to uniform -> nullifying C -> 1."""
    X, labels = _separable_data()
    zc_probe = _perfect_binary_probe(X.shape[1])
    ze_probe = _perfect_binary_probe(X.shape[1])
    vp = ValidationProbes(zc_probe=zc_probe, ze_probe=ze_probe,
                          acc_zc=1.0, acc_ze=1.0)

    X_post = torch.zeros_like(X)
    m = compute_tvd_metrics(vp, X_pre=X, X_post=X_post, zc=labels, ze=labels,
                            device=DEVICE)
    assert m.completeness_null == pytest.approx(1.0, abs=1e-5)


def test_compute_rejects_floor_at_or_below_chance():
    X, labels = _separable_data()
    vp = ValidationProbes(zc_probe=_perfect_binary_probe(X.shape[1]),
                          ze_probe=_perfect_binary_probe(X.shape[1]),
                          acc_zc=1.0, acc_ze=1.0)
    with pytest.raises(ValueError):
        compute_tvd_metrics(vp, X_pre=X, X_post=X, zc=labels, ze=labels,
                            device=DEVICE, chance=0.5, floor=0.5)


def test_tvdmetrics_as_dict_schema():
    m = TVDMetrics(completeness_null=0.9, completeness_cf=0.8, selectivity=0.7,
                   reliability_null=0.79, reliability_cf=0.74,
                   acc_zc_pre=0.99, acc_ze_pre=0.98,
                   floor=DECODABILITY_FLOOR, chance=0.5, mode="dist",
                   num_classes=2)
    d = m.as_dict()
    for key in ("C_null", "C_cf", "S", "R_null", "R_cf", "acc_zc_pre",
                "acc_ze_pre", "floor", "chance", "mode", "num_classes"):
        assert key in d, f"missing schema key {key}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
