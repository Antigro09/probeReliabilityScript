"""
Acceptance tests for the WS1 MKA fix (differentiable regularizer).

Root cause being guarded against: in v1, both MKA kernels were binary kNN
adjacencies built under torch.no_grad(), so the regularizer was
piecewise-constant with zero gradient a.e. and MKA training was
byte-identical to MLP training (1800/1800 seed-matched v1 records).

Tests:
    (a) autograd of the MKA term w.r.t. fc1 parameters has nonzero norm
    (b) seed-matched MKA vs MLP diverge in final parameters and accuracy
        beyond floating-point tolerance
    (c) lambda = 0 reproduces MLP exactly
    (d) identity hidden map yields near-max mka_score; a structure-destroying
        map scores far lower

Run with pytest, or directly: python -m tests.test_mka
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.probes import (
    MLPProbe, MKAProbe, ProbeTrainConfig, train_probe, probe_accuracy,
    knn_kernel, rbf_kernel, mka_score,
)
from src.repro import set_seed

DEVICE = torch.device("cpu")


def _synthetic_data(n: int = 512, d: int = 32, seed: int = 0):
    """Linearly separable-ish binary data with cluster structure."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    w = torch.randn(d, generator=g)
    y = (X @ w > 0).long()
    # Add label-correlated cluster offset so the manifold has structure.
    X = X + 0.5 * y.unsqueeze(1).float() * torch.randn(1, d, generator=g)
    return X, y


def _train_pair(mka_lambda: float, seed: int = 0, epochs: int = 8):
    """Train a seed-matched (MLP, MKA) pair on identical data.

    Mirrors the benchmark's seed handling: identical RNG state at probe
    construction (MLPProbe and MKAProbe build the same fc1/fc2 stack, so
    they consume the same RNG draws) and identical RNG state at training
    time (so mini-batch shuffles match).
    """
    X, y = _synthetic_data()
    dim, hidden = X.shape[1], 16
    cfg = ProbeTrainConfig(epochs=epochs, lr=1e-3, weight_decay=0.01,
                           batch_size=128)

    set_seed(seed)
    mlp = MLPProbe(dim, hidden_dim=hidden)
    set_seed(seed)
    mka = MKAProbe(dim, hidden_dim=hidden, mka_lambda=mka_lambda, knn_k=5)

    set_seed(seed + 1)
    train_probe(mlp, X, y, cfg, DEVICE)
    set_seed(seed + 1)
    train_probe(mka, X, y, cfg, DEVICE)
    return mlp, mka, X, y


# ---------------------------------------------------------------------------
# (a) The MKA term has nonzero gradient w.r.t. fc1 parameters
# ---------------------------------------------------------------------------

def test_mka_term_has_nonzero_gradient():
    set_seed(0)
    probe = MKAProbe(input_dim=20, hidden_dim=12, mka_lambda=1.0, knn_k=5)
    x = torch.randn(64, 20)
    _, hidden = probe(x, return_hidden=True)
    mka_term = probe.mka_loss(x, hidden)
    grads = torch.autograd.grad(mka_term, list(probe.fc1.parameters()),
                                allow_unused=True)
    assert grads[0] is not None, "MKA term is disconnected from fc1 weights"
    total_norm = sum(g.norm().item() for g in grads if g is not None)
    assert total_norm > 1e-6, (
        f"MKA regularizer gradient norm w.r.t. fc1 is {total_norm:.3e} — "
        f"the v1 zero-gradient bug is back."
    )
    assert torch.isfinite(torch.tensor(total_norm)), "MKA gradient is not finite"


def test_mka_gradient_finite_with_duplicate_points():
    """Coincident points must not produce NaN gradients (cdist pitfall)."""
    set_seed(0)
    probe = MKAProbe(input_dim=8, hidden_dim=6, mka_lambda=1.0, knn_k=3)
    x = torch.randn(16, 8)
    x[1] = x[0]  # exact duplicate
    _, hidden = probe(x, return_hidden=True)
    mka_term = probe.mka_loss(x, hidden)
    grads = torch.autograd.grad(mka_term, list(probe.fc1.parameters()))
    for g in grads:
        assert torch.isfinite(g).all(), "NaN/inf gradient at duplicate points"


# ---------------------------------------------------------------------------
# (b) Seed-matched MKA vs MLP diverge beyond fp tolerance
# ---------------------------------------------------------------------------

def test_seed_matched_mka_and_mlp_diverge():
    mlp, mka, X, y = _train_pair(mka_lambda=0.1)
    diffs = [(p1 - p2).abs().max().item()
             for p1, p2 in zip(mlp.parameters(), mka.parameters())]
    max_diff = max(diffs)
    assert max_diff > 1e-4, (
        f"Seed-matched MKA (lambda=0.1) and MLP parameters differ by at most "
        f"{max_diff:.3e} — regularizer is not shaping training."
    )
    acc_mlp = probe_accuracy(mlp, X, y, DEVICE)
    acc_mka = probe_accuracy(mka, X, y, DEVICE)
    assert abs(acc_mlp - acc_mka) > 0 or max_diff > 1e-3, (
        "MKA and MLP are numerically indistinguishable after training."
    )


# ---------------------------------------------------------------------------
# (c) lambda = 0 reproduces MLP exactly
# ---------------------------------------------------------------------------

def test_lambda_zero_reproduces_mlp_exactly():
    mlp, mka, _, _ = _train_pair(mka_lambda=0.0)
    for p1, p2 in zip(mlp.parameters(), mka.parameters()):
        assert torch.equal(p1, p2), (
            "MKA with lambda=0 must be byte-identical to seed-matched MLP "
            "(the ablation contract from the module docstring)."
        )


# ---------------------------------------------------------------------------
# (d) Identity hidden map yields near-max mka_score
# ---------------------------------------------------------------------------

def test_identity_hidden_map_near_max_score():
    """
    The score itself maxes out at 1.0 only for identical kernels (hard kNN
    vs soft RBF of the same points cannot coincide exactly), so "near-max"
    is operationalized two ways:
      1. exact self-alignment sanity: mka_score(K, K) == 1, and an identity
         hidden map scored with the SAME kernel family on both sides is
         exactly 1;
      2. against the mixed (hard-input, soft-hidden) pairing used in
         training, the identity map must clearly dominate a battery of
         structure-destroying hidden maps.
    """
    set_seed(0)
    X, _ = _synthetic_data(n=256, d=16)
    k = 5
    g = torch.Generator().manual_seed(1)

    K_in = knn_kernel(X, k=k)
    # Perfect self-alignment sanity checks.
    self_score = mka_score(K_in, K_in).item()
    assert self_score > 0.999, f"mka_score(K, K) = {self_score:.4f}, expected ~1"
    same_family = mka_score(rbf_kernel(X, k=k), rbf_kernel(X, k=k)).item()
    assert same_family > 0.999, (
        f"identity map, same kernel family both sides: {same_family:.4f}, expected ~1"
    )

    # Mixed pairing (what mka_loss actually computes): identity map vs a
    # battery of structure-destroying hidden maps.
    score_identity = mka_score(K_in, rbf_kernel(X, k=k)).item()
    alternatives = {
        "shuffled rows": X[torch.randperm(X.shape[0], generator=g)],
        "gaussian noise": torch.randn(X.shape[0], 16, generator=g),
        "random projection": X @ torch.randn(X.shape[1], 16, generator=g),
        "constant collapse": torch.ones(X.shape[0], 16),
    }
    scores_alt = {name: mka_score(K_in, rbf_kernel(h, k=k)).item()
                  for name, h in alternatives.items()}

    assert score_identity > 0.25, (
        f"Identity map scores only {score_identity:.3f} against its own "
        f"kNN kernel; the soft kernel is not tracking neighborhood structure."
    )
    for name, s in scores_alt.items():
        # Random projections preserve some geometry (Johnson-Lindenstrauss),
        # so require dominance, not obliteration.
        margin = 0.05 if name == "random projection" else 0.2
        assert score_identity > s + margin, (
            f"Identity map ({score_identity:.3f}) does not dominate "
            f"'{name}' ({s:.3f}); mka_score is not structure-sensitive."
        )


# ---------------------------------------------------------------------------
# Training smoke: soft-kernel MKA trains stably and stays accurate
# ---------------------------------------------------------------------------

def test_mka_training_stable_and_accurate():
    mlp, mka, X, y = _train_pair(mka_lambda=0.1, epochs=30)
    for p in mka.parameters():
        assert torch.isfinite(p).all(), "non-finite parameters after training"
    acc_mlp = probe_accuracy(mlp, X, y, DEVICE)
    acc_mka = probe_accuracy(mka, X, y, DEVICE)
    # The regularizer must shape training without tanking accuracy: stay
    # within a few points of the seed-matched unregularized baseline.
    assert acc_mka > acc_mlp - 0.05, (
        f"MKA (lambda=0.1) accuracy {acc_mka:.3f} vs seed-matched MLP "
        f"{acc_mlp:.3f}; regularizer is tanking learning."
    )


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
