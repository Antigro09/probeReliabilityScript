"""Tests for the WS5 repaired-reliability machinery (direction, projection, ladder)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.probes import LinearProbe
from src.metrics import ValidationProbes
from src.ws5_repaired import (
    candidate_direction, candidate_projection, candidate_edit,
    build_ladder, _mean_defined, _max_defined,
    _discriminant, _assert_not_clones, Evaluator, EvaluatorQualityError,
)

DEVICE = torch.device("cpu")


def _linear_probe_with_discriminant(D, u):
    """Build a LinearProbe whose (w[1]-w[0]) equals a target vector u."""
    p = LinearProbe(D)
    with torch.no_grad():
        p.linear.weight.zero_()
        p.linear.bias.zero_()
        p.linear.weight[1] = u                # w[1]-w[0] = u
    return p


def test_candidate_direction_linear_equals_discriminant():
    D = 12
    u = torch.randn(D)
    probe = _linear_probe_with_discriminant(D, u)
    X_ref = torch.randn(50, D)
    Q = candidate_direction(probe, X_ref, DEVICE, r=1)
    assert Q.shape == (D, 1)
    # top singular vector must align with normalize(u) up to sign
    cos = torch.dot(Q[:, 0], u / u.norm()).abs()
    assert float(cos) == pytest.approx(1.0, abs=1e-4)


def test_candidate_direction_degenerate_returns_none():
    D = 8
    probe = _linear_probe_with_discriminant(D, torch.zeros(D))  # flat logit-diff
    X_ref = torch.randn(30, D)
    assert candidate_direction(probe, X_ref, DEVICE) is None


def test_candidate_projection_properties():
    D = 10
    Q = torch.linalg.qr(torch.randn(D, 2))[0][:, :2]
    P = candidate_projection(Q)
    # symmetric, idempotent, annihilates Q
    assert torch.allclose(P, P.T, atol=1e-5)
    assert torch.allclose(P @ P, P, atol=1e-4)
    assert torch.allclose(P @ Q, torch.zeros(D, 2), atol=1e-4)


def test_candidate_edit_with_override():
    D = 6
    probe = _linear_probe_with_discriminant(D, torch.randn(D))
    X_inter = torch.randn(20, D)
    override = torch.zeros(D); override[0] = 1.0
    edit = candidate_edit(probe, X_inter, X_inter, DEVICE, d_override=override)
    # the first coordinate is erased
    assert edit[:, 0].abs().max() < 1e-4


def test_build_ladder_mean_not_max():
    per_method = {
        "INLP": {"R": 0.4}, "RLACE": {"R": 0.6},
        "AlterRep": {"R": 0.9}, "FGSM": {"R": 0.9}, "PGD": {"R": 0.9},
        "cand": {"R": 0.5},
    }
    lad = build_ladder(per_method)
    assert lad["R_v1_max"] == pytest.approx(0.9)
    assert lad["R_v1_mean"] == pytest.approx((0.4 + 0.6 + 0.9 + 0.9 + 0.9) / 5)
    assert lad["R_excl"] == pytest.approx(0.5)      # mean of INLP,RLACE
    assert lad["R_cand"] == pytest.approx(0.5)
    assert lad["R_rep"] == lad["R_cand"]


def test_ladder_handles_none():
    per_method = {"INLP": {"R": None}, "RLACE": {"R": 0.3}, "cand": {"R": None}}
    lad = build_ladder(per_method)
    assert lad["R_excl"] == pytest.approx(0.3)      # None skipped
    assert lad["R_cand"] is None
    assert lad["R_v1_max"] == pytest.approx(0.3)


def test_mean_max_defined_skip_none():
    assert _mean_defined([None, 0.2, 0.4]) == pytest.approx(0.3)
    assert _max_defined([None, 0.2, 0.4]) == pytest.approx(0.4)
    assert _mean_defined([None, None]) is None


def test_clone_detection_raises():
    D = 8
    u = torch.randn(D)
    vp = ValidationProbes(_linear_probe_with_discriminant(D, u),
                          _linear_probe_with_discriminant(D, u), 1.0, 1.0)
    clones = [Evaluator(vp, s, 0.99, 0.99) for s in range(3)]  # identical discriminants
    with pytest.raises(EvaluatorQualityError):
        _assert_not_clones(clones, clone_cos_max=1.0 - 1e-4)


def test_discriminant_unit_norm():
    D = 5
    u = torch.randn(D)
    vp = ValidationProbes(_linear_probe_with_discriminant(D, u),
                          _linear_probe_with_discriminant(D, u), 1.0, 1.0)
    d = _discriminant(vp)
    assert float(d.norm()) == pytest.approx(1.0, abs=1e-5)
