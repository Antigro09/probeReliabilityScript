"""
Tests for WS4: the attacker/evaluator completeness control.

WHY
---
WS4 claims that when a direction-based intervention (AlterRep / FGSM / PGD)
is BOTH generated from and scored by the same probe (MATCHED), completeness
is inflated relative to scoring with an INDEPENDENT evaluator probe (SPLIT).
The core assertion these tests pin down is directional:

    matched_C >= split_C - tolerance

i.e. matched completeness is not LOWER than split completeness for the
direction-based methods. Equality (gap ~ 0) is allowed; the confound may be
weak on any given synthetic draw, but the matched edit — which targets the
scorer's own boundary — should never score as LESS complete than when a
foreign probe reads it off.

These tests run on synthetic gaussian representations: zc is linearly
separable along a planted axis, ze along an orthogonal axis. No model, no
cache, no GPU — the pure function attacker_evaluator_completeness is
model-free by design. The full CLI path (cache discovery + model subset) is
NOT unit-tested here; it needs a real benchmark cache on a GPU machine.

Requires torch + numpy (installed on the GPU test host). Marked so a
torch-less environment skips cleanly rather than erroring at collection.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

# Dependency-light guard: skip the whole module if torch is unavailable so
# collection never hard-fails on a machine without the ML stack.
_HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(
    not _HAS_TORCH, reason="torch not installed in this environment"
)

if _HAS_TORCH:
    import torch

    from src.probes import ProbeTrainConfig
    from src.interventions import InterventionConfig
    from scripts.ws4_attacker_evaluator import (
        attacker_evaluator_completeness,
        _three_way_split,
        _bootstrap_ci,
        DIRECTION_METHODS,
        CONTROL_METHODS,
    )


# ---------------------------------------------------------------------------
# Synthetic data: zc separable along axis 0, ze along axis 1 (orthogonal).
# ---------------------------------------------------------------------------

def _make_synthetic(n: int = 900, d: int = 16, sep: float = 4.0, seed: int = 0):
    """
    Gaussian reps with a planted, decodable structure:
        - zc (target)   is separable along basis axis 0
        - ze (attractor) is separable along basis axis 1 (orthogonal to 0)
    Labels are balanced and independent of each other. `sep` is the
    class-mean gap; large enough that a linear probe is comfortably above
    the decodability floor on clean reps.
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    zc = torch.randint(0, 2, (n,), generator=g)
    ze = torch.randint(0, 2, (n,), generator=g)
    # Shift class means apart along orthogonal axes.
    X[:, 0] += sep * (zc.float() - 0.5)
    X[:, 1] += sep * (ze.float() - 0.5)
    return X.float(), zc.long(), ze.long()


def _fast_cfgs():
    """Cheap probe/intervention configs so the test runs quickly on GPU/CPU."""
    cfg = ProbeTrainConfig(epochs=15, lr=1e-2, weight_decay=0.0, batch_size=128)
    inter_cfg = InterventionConfig(
        inlp_iters=3, rlace_rank=1, rlace_steps=30,
        alterrep_alpha=3.0, fgsm_eps=3.0,
        pgd_eps=3.0, pgd_steps=5, pgd_alpha=1.0,
    )
    return cfg, inter_cfg


# ---------------------------------------------------------------------------
# Fold-splitting: disjoint and exhaustive, reproducible from (n, seed).
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_three_way_split_disjoint_and_exhaustive():
    n = 301
    a, e, i = _three_way_split(n, seed=7)
    all_idx = torch.cat([a, e, i])
    assert all_idx.numel() == n
    assert torch.unique(all_idx).numel() == n  # no overlaps, no gaps
    # Deterministic from (n, seed).
    a2, e2, i2 = _three_way_split(n, seed=7)
    assert torch.equal(a, a2) and torch.equal(e, e2) and torch.equal(i, i2)
    # A different seed permutes differently.
    a3, _, _ = _three_way_split(n, seed=8)
    assert not torch.equal(a, a3)


# ---------------------------------------------------------------------------
# Bootstrap CI: ordering and degenerate cases.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_bootstrap_ci_ordering_and_degenerate():
    import numpy as np
    rng = np.random.default_rng(0)
    vals = [0.2, 0.4, 0.6, 0.8, 0.5]
    lo, hi = _bootstrap_ci(vals, rng, n_boot=500)
    assert lo is not None and hi is not None
    assert lo <= np.mean(vals) <= hi
    # Single value -> degenerate interval at that value.
    lo1, hi1 = _bootstrap_ci([0.7], rng, n_boot=500)
    assert lo1 == pytest.approx(0.7) and hi1 == pytest.approx(0.7)
    # All-None -> (None, None).
    lo0, hi0 = _bootstrap_ci([None, None], rng, n_boot=500)
    assert lo0 is None and hi0 is None


# ---------------------------------------------------------------------------
# Output dict shape: methods, controls, CI ordering.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_output_shape_and_controls_present():
    X, zc, ze = _make_synthetic()
    cfg, inter_cfg = _fast_cfgs()
    device = torch.device("cpu")
    out = attacker_evaluator_completeness(
        X, zc, ze, device,
        M=3, cfg=cfg, inter_cfg=inter_cfg,
        methods=DIRECTION_METHODS, controls=CONTROL_METHODS,
        bootstrap=300, seed=0,
    )

    # Top-level keys.
    for key in ("methods", "controls", "M", "bootstrap", "seed", "floor",
                "n_attacker", "n_evaluator", "n_intervention"):
        assert key in out, f"missing top-level key {key}"

    # Every requested direction method present with the full schema.
    for method in DIRECTION_METHODS:
        assert method in out["methods"]
        mo = out["methods"][method]
        for key in ("matched_C_mean", "matched_C_ci", "matched_n",
                    "split_C_mean", "split_C_ci", "split_n", "gap",
                    "matched_C_pairs", "split_C_pairs"):
            assert key in mo, f"{method} missing {key}"
        assert len(mo["matched_C_pairs"]) == 3
        assert len(mo["split_C_pairs"]) == 3

    # Controls present (baselines).
    for control in CONTROL_METHODS:
        assert control in out["controls"], f"missing control {control}"
        co = out["controls"][control]
        for key in ("C_mean", "C_ci", "n", "C_pairs"):
            assert key in co, f"control {control} missing {key}"


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_ci_ordering_lo_le_mean_le_hi():
    X, zc, ze = _make_synthetic()
    cfg, inter_cfg = _fast_cfgs()
    device = torch.device("cpu")
    out = attacker_evaluator_completeness(
        X, zc, ze, device,
        M=4, cfg=cfg, inter_cfg=inter_cfg,
        bootstrap=400, seed=1,
    )
    for method, mo in out["methods"].items():
        for cond in ("matched", "split"):
            mean = mo[f"{cond}_C_mean"]
            lo, hi = mo[f"{cond}_C_ci"]
            if mean is None:
                continue  # every pair below floor: CI may be (None, None)
            assert lo is not None and hi is not None
            assert lo <= mean + 1e-9, f"{method}/{cond}: lo>{mean}"
            assert mean <= hi + 1e-9, f"{method}/{cond}: mean>{hi}"
    for control, co in out["controls"].items():
        mean = co["C_mean"]
        lo, hi = co["C_ci"]
        if mean is None:
            continue
        assert lo <= mean + 1e-9 and mean <= hi + 1e-9


# ---------------------------------------------------------------------------
# The core scientific assertion: matched_C >= split_C - tolerance.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_matched_completeness_not_lower_than_split():
    """
    For each direction-based method, matched completeness must not be LOWER
    than split completeness (up to a small tolerance). The matched edit
    targets the scorer's own boundary, so it should erase at least as much
    of the scorer-readable Zc signal as a foreign evaluator sees.
    """
    # tolerance absorbs probe-training noise on a small synthetic draw; it is
    # a test-only slack, not an experimenter threshold.
    TOL = 0.10
    X, zc, ze = _make_synthetic(n=1200, sep=4.0, seed=3)
    cfg, inter_cfg = _fast_cfgs()
    device = torch.device("cpu")
    out = attacker_evaluator_completeness(
        X, zc, ze, device,
        M=5, cfg=cfg, inter_cfg=inter_cfg,
        methods=DIRECTION_METHODS, controls=(),  # controls not needed here
        bootstrap=200, seed=3,
    )
    for method, mo in out["methods"].items():
        matched = mo["matched_C_mean"]
        split = mo["split_C_mean"]
        # On healthy synthetic reps both should be decodable (non-null).
        assert matched is not None, f"{method}: matched_C unexpectedly null"
        assert split is not None, f"{method}: split_C unexpectedly null"
        assert matched >= split - TOL, (
            f"{method}: matched_C ({matched:.3f}) is more than {TOL} below "
            f"split_C ({split:.3f}) — the attacker/evaluator inflation should "
            f"never make matched completeness LOWER than split."
        )


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_gap_matches_mean_difference():
    """gap must equal matched_C_mean - split_C_mean when both are defined."""
    X, zc, ze = _make_synthetic(seed=5)
    cfg, inter_cfg = _fast_cfgs()
    device = torch.device("cpu")
    out = attacker_evaluator_completeness(
        X, zc, ze, device,
        M=3, cfg=cfg, inter_cfg=inter_cfg, bootstrap=200, seed=5,
    )
    for method, mo in out["methods"].items():
        m, s, gap = mo["matched_C_mean"], mo["split_C_mean"], mo["gap"]
        if m is None or s is None:
            assert gap is None
        else:
            assert gap == pytest.approx(m - s)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
