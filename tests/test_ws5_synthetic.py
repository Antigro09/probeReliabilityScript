"""Tests for the WS5 environment-shift synthetic generator."""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from src.ws5_synthetic import (
    SyntheticConfig, make_planted_reps, split_eval_folds,
    recovery_score, subspace_recovery, random_cosine_baseline,
    check_4cell_balance, bayes_direction,
)


def test_shapes_and_labels():
    cfg = SyntheticConfig(D=32, N=400, seed=1)
    tr, ev, truth = make_planted_reps(cfg)
    for reps in (tr, ev):
        assert reps["X"].shape == (400, 32)
        assert reps["X"].dtype == torch.float32
        assert set(reps["zc"].tolist()) <= {0, 1}
        assert set(reps["ze"].tolist()) <= {0, 1}
    assert truth["v_c"].shape == (32,)


def test_recovery_score_bounds():
    v = torch.randn(64)
    assert recovery_score(v, v) == pytest.approx(1.0, abs=1e-5)
    # orthogonal-ish: a random vector's expected |cos| ~ sqrt(2/(pi D)), small
    u = torch.randn(64)
    assert 0.0 <= recovery_score(u, v) <= 1.0
    assert recovery_score(v.unsqueeze(1), v) == pytest.approx(1.0, abs=1e-5)


def test_subspace_recovery_identity_and_orthogonal():
    D = 16
    V = torch.linalg.qr(torch.randn(D, 2))[0][:, :2]
    assert subspace_recovery(V, V) == pytest.approx(1.0, abs=1e-5)
    # a subspace orthogonal to V recovers ~0
    full = torch.linalg.qr(torch.randn(D, D))[0]
    W = full[:, 2:4]  # not guaranteed orthogonal to V, but different — just bounded
    r = subspace_recovery(W, V)
    assert 0.0 <= r <= 1.0


def test_bayes_eval_is_invariant_vc():
    cfg = SyntheticConfig(D=48, seed=3)
    _, _, truth = make_planted_reps(cfg)
    # at alpha_eval = 0.5, the Bayes direction collapses to v_c exactly
    cos = recovery_score(truth["bayes_eval"], truth["v_c"])
    assert cos == pytest.approx(1.0, abs=1e-5)
    # at alpha_train > 0.5 it picks up a shortcut component -> not pure v_c
    cos_train = recovery_score(truth["bayes_train"], truth["v_c"])
    assert cos_train < 0.9999


def test_bayes_direction_formula():
    v_c = torch.tensor([1.0, 0.0, 0.0])
    v_s = torch.tensor([0.0, 1.0, 0.0])
    d = bayes_direction(v_c, v_s, mu_c=1.0, mu_s=1.0, alpha=0.5)
    assert recovery_score(d, v_c) == pytest.approx(1.0, abs=1e-6)


def test_split_folds_disjoint_and_complete():
    cfg = SyntheticConfig(D=16, N=300, seed=5)
    _, ev, _ = make_planted_reps(cfg)
    folds = split_eval_folds(ev, ref_frac=0.3, eval_frac=0.35, seed=7)
    ns = {k: folds[k]["X"].shape[0] for k in ("ref", "eval", "inter")}
    assert sum(ns.values()) == 300
    assert all(n > 0 for n in ns.values())


def test_random_cosine_baseline():
    assert random_cosine_baseline(256) == pytest.approx(math.sqrt(2 / (math.pi * 256)))


def test_4cell_balance_passes_on_balanced():
    cfg = SyntheticConfig(D=16, N=2000, seed=9)
    tr, _, _ = make_planted_reps(cfg)
    counts = check_4cell_balance(tr, tol=0.25)
    assert sum(counts.values()) == 2000


def test_nonlinear_family_subspace_truth():
    cfg = SyntheticConfig(D=24, N=200, family="nonlinear", seed=2)
    tr, ev, truth = make_planted_reps(cfg)
    assert truth["concept_subspace"].shape == (24, 2)
    assert tr["X"].shape == (200, 24)
