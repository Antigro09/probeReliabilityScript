"""
Tests for WS6: honest v2 statistics (scripts/paper_stats.py) and the locked
v2 evaluator's shared building blocks.

The four functions under test each fix a specific dishonesty in the v1
exploratory analysis:

  * tie_aware_hit_rate  — replaces the list-order max (predictor_eval.py:
    139-140) with an overlap fraction, and reports how often ties occurred.
  * distinct_r_report   — surfaces that v1/v2 R is identical across the
    archs×seeds inside a cell (distinct-R == 1), i.e. the target has no
    per-probe variance.
  * cluster_bootstrap_spearman — resamples whole (model,layer,task) clusters;
    must be deterministic under a fixed seed.
  * permutation_spearman — distribution-free p; must lie in [0, 1].

These tests are torch-free by design (the functions are pure Python / numpy /
scipy). numpy and scipy ARE required because scripts/paper_stats imports them
at module load; the tests are skipped cleanly if either is unavailable so the
suite still collects on a bare machine.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

# scripts/paper_stats imports numpy + scipy at module load. Skip the whole
# module (rather than erroring at collection) when they are absent.
np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from scripts.paper_stats import (
    cluster_bootstrap_spearman,
    permutation_spearman,
    tie_aware_hit_rate,
    distinct_r_report,
)


# ---------------------------------------------------------------------------
# Helpers to build toy records / cells.
# ---------------------------------------------------------------------------

def _rec(model, layer, task, arch, seed, A, R):
    """Minimal benchmark record with just the keys the WS6 helpers read."""
    return {"model": model, "layer": layer, "task": task, "arch": arch,
            "seed": seed, "A": A, "R": R}


# ---------------------------------------------------------------------------
# tie_aware_hit_rate
# ---------------------------------------------------------------------------

def test_tie_aware_all_equal_gives_full_tie_and_fractional_hit():
    """Every arch ties for max A AND max R -> tie_fraction 1.0, hit == 1.0.

    With all three archs tied on both sides, argmaxA == argmaxR == {all 3}, so
    the overlap fraction is 3/3 = 1.0 for the cell. Crucially it is NOT decided
    by list order (the v1 bug), and the tie is reported.
    """
    cells = {
        ("m", 0, "sva"): [
            {"arch": "linear", "A": 0.5, "R": 0.7},
            {"arch": "mlp", "A": 0.5, "R": 0.7},
            {"arch": "mka", "A": 0.5, "R": 0.7},
        ]
    }
    out = tie_aware_hit_rate(cells)
    assert out["total"] == 1
    assert out["tie_fraction"] == pytest.approx(1.0)
    # |argmaxA ∩ argmaxR| / |argmaxA| = 3/3
    assert out["hits"] == pytest.approx(1.0)
    assert out["hit_rate"] == pytest.approx(1.0)


def test_tie_aware_unique_best_matching():
    """Unique best-A arch is also the unique best-R arch -> clean hit, no tie."""
    cells = {
        ("m", 0, "sva"): [
            {"arch": "linear", "A": 0.9, "R": 0.9},
            {"arch": "mlp", "A": 0.5, "R": 0.4},
            {"arch": "mka", "A": 0.3, "R": 0.2},
        ]
    }
    out = tie_aware_hit_rate(cells)
    assert out["hits"] == pytest.approx(1.0)
    assert out["hit_rate"] == pytest.approx(1.0)
    assert out["tie_fraction"] == pytest.approx(0.0)


def test_tie_aware_unique_best_mismatching():
    """Best A and best R are different unique archs -> miss (hit 0)."""
    cells = {
        ("m", 0, "sva"): [
            {"arch": "linear", "A": 0.9, "R": 0.1},
            {"arch": "mlp", "A": 0.2, "R": 0.9},
        ]
    }
    out = tie_aware_hit_rate(cells)
    assert out["hits"] == pytest.approx(0.0)
    assert out["hit_rate"] == pytest.approx(0.0)
    assert out["tie_fraction"] == pytest.approx(0.0)


def test_tie_aware_partial_overlap_fraction():
    """Two archs tie for best A; only one of them is the best-R arch.

    argmaxA = {linear, mlp}, argmaxR = {linear}. Overlap 1/2 -> hit 0.5, and a
    tie occurred (on the A side).
    """
    cells = {
        ("m", 0, "sva"): [
            {"arch": "linear", "A": 0.8, "R": 0.9},
            {"arch": "mlp", "A": 0.8, "R": 0.3},
            {"arch": "mka", "A": 0.1, "R": 0.2},
        ]
    }
    out = tie_aware_hit_rate(cells)
    assert out["hits"] == pytest.approx(0.5)
    assert out["hit_rate"] == pytest.approx(0.5)
    assert out["tie_fraction"] == pytest.approx(1.0)


def test_tie_aware_skips_single_arch_and_all_null_R():
    """Cells with < 2 archs, or with R null everywhere, are skipped."""
    cells = {
        ("m", 0, "sva"): [{"arch": "linear", "A": 0.5, "R": 0.5}],   # 1 arch
        ("m", 1, "sva"): [                                            # all null R
            {"arch": "linear", "A": 0.5, "R": None},
            {"arch": "mlp", "A": 0.6, "R": None},
        ],
    }
    out = tie_aware_hit_rate(cells)
    assert out["total"] == 0
    assert out["n_skipped"] == 2
    assert out["hit_rate"] == 0.0
    assert out["tie_fraction"] == 0.0


def test_tie_aware_ignores_null_R_but_ranks_the_rest():
    """A null-R arch is ignored on the R side but the cell is still ranked."""
    cells = {
        ("m", 0, "sva"): [
            {"arch": "linear", "A": 0.9, "R": None},   # best A, but no R
            {"arch": "mlp", "A": 0.4, "R": 0.7},        # only decodable R
        ]
    }
    out = tie_aware_hit_rate(cells)
    # argmaxA = {linear}, argmaxR = {mlp} -> no overlap -> miss, but ranked.
    assert out["total"] == 1
    assert out["hits"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# distinct_r_report
# ---------------------------------------------------------------------------

def test_distinct_r_constant_across_archs_is_one():
    """R identical across archs×seeds in a cell -> distinct == 1 (the defect)."""
    recs = []
    for arch in ("linear", "mlp", "mka"):
        for seed in (1000, 1001):
            recs.append(_rec("m", 0, "sva", arch, seed, A=0.3, R=0.42))
    out = distinct_r_report(recs)
    assert out["n_cells"] == 1
    assert out["per_cell"][("m", 0, "sva")]["distinct"] == 1
    assert out["per_cell"][("m", 0, "sva")]["n_rows"] == 6
    assert out["histogram"] == {1: 1}
    assert out["frac_degenerate"] == pytest.approx(1.0)


def test_distinct_r_varying_counts_distinct():
    """A cell where R genuinely varies across archs -> distinct > 1."""
    recs = [
        _rec("m", 0, "sva", "linear", 1000, 0.3, 0.42),
        _rec("m", 0, "sva", "mlp", 1000, 0.3, 0.55),
        _rec("m", 0, "sva", "mka", 1000, 0.3, 0.61),
    ]
    out = distinct_r_report(recs)
    assert out["per_cell"][("m", 0, "sva")]["distinct"] == 3
    assert out["histogram"] == {3: 1}
    assert out["frac_degenerate"] == pytest.approx(0.0)


def test_distinct_r_two_cells_mixed():
    """One degenerate cell, one varying cell -> histogram {1:1, 2:1}."""
    recs = [
        # cell A: constant R
        _rec("m", 0, "sva", "linear", 1000, 0.3, 0.5),
        _rec("m", 0, "sva", "mlp", 1000, 0.3, 0.5),
        # cell B: two distinct R
        _rec("m", 1, "sva", "linear", 1000, 0.3, 0.4),
        _rec("m", 1, "sva", "mlp", 1000, 0.3, 0.6),
    ]
    out = distinct_r_report(recs)
    assert out["n_cells"] == 2
    assert out["histogram"] == {1: 1, 2: 1}
    assert out["frac_degenerate"] == pytest.approx(0.5)


def test_distinct_r_null_counts_as_one_category():
    """All-null R in a cell reads as distinct == 1 (not 0), with n_null tracked."""
    recs = [
        _rec("m", 0, "sva", "linear", 1000, 0.3, None),
        _rec("m", 0, "sva", "mlp", 1000, 0.3, None),
    ]
    out = distinct_r_report(recs)
    cell = out["per_cell"][("m", 0, "sva")]
    assert cell["distinct"] == 1
    assert cell["n_null"] == 2
    assert out["frac_degenerate"] == pytest.approx(1.0)


def test_distinct_r_null_plus_value_is_two_categories():
    """None and a real value are two distinct categories."""
    recs = [
        _rec("m", 0, "sva", "linear", 1000, 0.3, None),
        _rec("m", 0, "sva", "mlp", 1000, 0.3, 0.7),
    ]
    out = distinct_r_report(recs)
    cell = out["per_cell"][("m", 0, "sva")]
    assert cell["distinct"] == 2
    assert cell["n_null"] == 1


def test_distinct_r_prefers_floored_key_when_present():
    """When _R_floored is present it, not the legacy R, is counted."""
    recs = [
        {"model": "m", "layer": 0, "task": "sva", "arch": "linear",
         "seed": 1000, "A": 0.3, "R": 0.9, "_R_floored": None},
        {"model": "m", "layer": 0, "task": "sva", "arch": "mlp",
         "seed": 1000, "A": 0.3, "R": 0.9, "_R_floored": None},
    ]
    out = distinct_r_report(recs)
    cell = out["per_cell"][("m", 0, "sva")]
    # Floored R is None for both -> distinct 1, n_null 2 (legacy R=0.9 ignored).
    assert cell["distinct"] == 1
    assert cell["n_null"] == 2


def test_distinct_r_rounding_ignores_fp_noise():
    """Values differing only by <1e-9 collapse to one distinct value."""
    recs = [
        _rec("m", 0, "sva", "linear", 1000, 0.3, 0.5),
        _rec("m", 0, "sva", "mlp", 1000, 0.3, 0.5 + 1e-12),
    ]
    out = distinct_r_report(recs)
    assert out["per_cell"][("m", 0, "sva")]["distinct"] == 1


# ---------------------------------------------------------------------------
# cluster_bootstrap_spearman
# ---------------------------------------------------------------------------

def _clustered_pairs(n_clusters=12, per_cluster=3, seed=1):
    """Build (pairs, cluster_ids) with a real A-R relationship.

    R is constant within a cluster (mirrors the shared-R structure of the real
    data); A varies within a cluster and correlates with the cluster's R.
    """
    rng = np.random.default_rng(seed)
    pairs, cluster_ids = [], []
    for c in range(n_clusters):
        base_r = c / n_clusters
        for _ in range(per_cluster):
            a = base_r + rng.normal(0, 0.05)
            pairs.append((a, base_r))
            cluster_ids.append(("m", c, "sva"))
    return pairs, cluster_ids


def test_cluster_bootstrap_is_deterministic_under_fixed_seed():
    pairs, cluster_ids = _clustered_pairs()
    a = cluster_bootstrap_spearman(pairs, cluster_ids, B=500, seed=0)
    b = cluster_bootstrap_spearman(pairs, cluster_ids, B=500, seed=0)
    assert a == b
    assert a["rho"] == pytest.approx(b["rho"])
    assert a["ci_low"] == pytest.approx(b["ci_low"])
    assert a["ci_high"] == pytest.approx(b["ci_high"])


def test_cluster_bootstrap_seed_changes_ci():
    """Different seeds give (generally) different CI bounds."""
    pairs, cluster_ids = _clustered_pairs()
    a = cluster_bootstrap_spearman(pairs, cluster_ids, B=500, seed=0)
    b = cluster_bootstrap_spearman(pairs, cluster_ids, B=500, seed=1)
    # Point estimate is seed-independent; the CI resample is not.
    assert a["rho"] == pytest.approx(b["rho"])
    assert (a["ci_low"], a["ci_high"]) != (b["ci_low"], b["ci_high"])


def test_cluster_bootstrap_ci_is_valid_ordered_interval():
    pairs, cluster_ids = _clustered_pairs()
    out = cluster_bootstrap_spearman(pairs, cluster_ids, B=1000, seed=0)
    assert out["n_clusters"] == 12
    # A percentile bootstrap CI need NOT bracket the point estimate (esp. when
    # rho sits near its ceiling), but it must be a valid ordered interval whose
    # bounds are legal Spearman values.
    assert -1.0 <= out["ci_low"] <= out["ci_high"] <= 1.0
    # Positive constructed relationship should show up in both point and CI.
    assert out["rho"] > 0.0
    assert out["ci_high"] > 0.0


def test_cluster_bootstrap_single_cluster_returns_nan_ci():
    pairs = [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]
    cluster_ids = [("m", 0, "sva")] * 3
    out = cluster_bootstrap_spearman(pairs, cluster_ids, B=100, seed=0)
    assert out["n_clusters"] == 1
    assert np.isnan(out["ci_low"])
    assert np.isnan(out["ci_high"])


def test_cluster_bootstrap_length_mismatch_raises():
    with pytest.raises(ValueError):
        cluster_bootstrap_spearman([(0.1, 0.2)], [("m", 0, "sva"), ("m", 1, "sva")])


# ---------------------------------------------------------------------------
# permutation_spearman
# ---------------------------------------------------------------------------

def test_permutation_p_in_unit_interval():
    rng = np.random.default_rng(0)
    A = rng.normal(size=40)
    R = A + rng.normal(0, 0.5, size=40)   # correlated
    out = permutation_spearman(A, R, n_perm=1000, seed=0)
    assert 0.0 <= out["p"] <= 1.0
    assert 0.0 <= out["p_one_sided"] <= 1.0
    assert out["n"] == 40


def test_permutation_strong_signal_is_significant():
    rng = np.random.default_rng(2)
    A = np.arange(50, dtype=float)
    R = A + rng.normal(0, 0.01, size=50)  # near-perfect monotone
    out = permutation_spearman(A, R, n_perm=2000, seed=0)
    assert out["rho"] > 0.99
    # add-one correction: smallest achievable p is 1/(n_perm+1); should be tiny.
    assert out["p"] < 0.01


def test_permutation_null_signal_p_not_tiny():
    rng = np.random.default_rng(3)
    A = rng.normal(size=60)
    R = rng.normal(size=60)               # independent
    out = permutation_spearman(A, R, n_perm=2000, seed=0)
    assert out["p"] > 0.05


def test_permutation_determinism():
    rng = np.random.default_rng(4)
    A = rng.normal(size=30)
    R = rng.normal(size=30)
    a = permutation_spearman(A, R, n_perm=500, seed=7)
    b = permutation_spearman(A, R, n_perm=500, seed=7)
    assert a == b


def test_permutation_length_mismatch_raises():
    with pytest.raises(ValueError):
        permutation_spearman([0.1, 0.2, 0.3], [0.1, 0.2])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
