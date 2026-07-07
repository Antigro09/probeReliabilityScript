"""Exploratory descriptive statistics for the paper write-up.

NOT part of the pre-registered analysis (that is scripts/predictor_eval.py).
This script aggregates benchmark records into per-model, per-task,
per-architecture, and per-layer-depth tables for reporting.

The input directory is explicit (WS0.2): v1 and v2 records must never
blend invisibly into one table. When pointed at v2 records, the LEGACY
clipped reliability (R_legacy) is used for every row — uniform v1
semantics, announced loudly — because this script's aggregations predate
the floored metrics. Floored (nullable-R) analysis ships with the v2
stats tooling (WS6).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# WS6 v2 "honest stats" constants.
#
# These control the additive '== v2 honest stats ==' section only; the
# exploratory output above them is byte-for-byte unchanged. Any value here
# that is an experimenter choice (not a mechanical default) is flagged so it
# can be migrated into the locked pre-registration.
# --------------------------------------------------------------------------
# Numerical tolerance for "tied for the max". Two values within this of the
# running max are both counted as arg-max members. Mechanical default, not an
# experimenter choice, but the exact value shapes tie_fraction:
TIE_TOL = 1e-9
# Default resample counts for the bootstrap / permutation helpers. These are
# analysis-power choices (more resamples = tighter CI / finer p-grid). If the
# v2 pre-registration fixes them, lock in PREREGISTRATION_v2.md.
BOOTSTRAP_B = 2000            # lock in PREREGISTRATION_v2.md
PERMUTATION_N = 10000         # lock in PREREGISTRATION_v2.md


def load_records(bench_dir: Path):
    recs = []
    for f in sorted(bench_dir.glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    recs.append(json.loads(line))
    if not recs:
        sys.exit(f"no *.jsonl records found in {bench_dir}")

    versions = {r.get("schema_version", 1) for r in recs}
    if len(versions) > 1:
        sys.exit(f"{bench_dir} mixes schema versions {sorted(versions)}; "
                 f"keep record versions in separate directories.")
    if versions == {2}:
        print("=" * 70)
        print("  NOTE: v2 records — using LEGACY clipped reliability "
              "(R_legacy) for\n  every row so aggregations stay uniform and "
              "v1-comparable. Floored\n  (nullable) R analysis belongs to "
              "the v2 stats tooling (WS6).")
        print("=" * 70)
        for r in recs:
            # Preserve the honest floored/nullable R (additive, WS6) BEFORE the
            # legacy overwrite so the v2 stats section can diagnose it. This
            # does not touch any existing key or output line.
            r["_R_floored"] = r["R"]
            r["R"] = r["R_legacy"]
            r["R_method"] = r["R_legacy_method"]
    return recs


def cell_medians(recs):
    """Replicate predictor_eval aggregation: median over seeds per
    (model, layer, task, arch), giving 270 (A, R) pairs."""
    groups = defaultdict(lambda: {"A": [], "R": []})
    for r in recs:
        key = (r["model"], r["layer"], r["task"], r["arch"])
        groups[key]["A"].append(r["A"])
        groups[key]["R"].append(r["R"])
    return {k: (statistics.median(v["A"]), statistics.median(v["R"]))
            for k, v in groups.items()}


def hit_rate(cells, keys):
    """Rank-1 hit rate over (model, layer, task) cells restricted to keys."""
    by_cell = defaultdict(dict)
    for (model, layer, task, arch), (a, r) in cells.items():
        if (model, layer, task) in keys:
            by_cell[(model, layer, task)][arch] = (a, r)
    hits = total = 0
    for archs in by_cell.values():
        best_a = max(archs, key=lambda k: archs[k][0])
        best_r = max(archs, key=lambda k: archs[k][1])
        hits += int(best_a == best_r)
        total += 1
    return hits, total


# ==========================================================================
# WS6 — honest v2 statistics
#
# WHY these exist: the exploratory helpers above treat every (A, R) pair as an
# independent observation and pick arg-max by Python list order. Neither is
# defensible for the headline claim. Three problems, three fixes:
#
#   1. Pseudo-replication. R is computed once per (model, layer, task) cell
#      from the layer's validation+data probes; it is IDENTICAL across the
#      archs×seeds inside that cell (see run_benchmark.run_one_cell — per_method
#      and hence R depend only on the pre-computed val_probes/interventions,
#      not on the evaluated probe). So the 270 "pairs" are really ~cell-many
#      independent draws. A naive Spearman p-value is anticonservative.
#      cluster_bootstrap_spearman resamples WHOLE clusters.
#
#   2. Tie-blind arg-max. predictor_eval.evaluate_p2 (lines 139-140) uses
#      `max(cells, key=...)`, which returns the FIRST arg-max on ties. When
#      several archs share the top A (or top R) — common once R is quantised
#      per cell — that silently rewards list order. tie_aware_hit_rate scores
#      the overlap fraction |argmaxA ∩ argmaxR| / |argmaxA| and reports how
#      often ties happened.
#
#   3. Hidden degeneracy. If R never varies across archs/seeds within a cell,
#      the whole "which probe is most reliable?" question is vacuous for that
#      cell. distinct_r_report surfaces that directly: a cell with
#      distinct-R == 1 means R did not depend on the probe at all.
# ==========================================================================


def _spearman_rho(a, b) -> float:
    """Spearman rho as a plain float; nan when undefined (constant input)."""
    if len(a) < 2:
        return float("nan")
    rho, _ = spearmanr(a, b)
    return float(rho)


def cluster_bootstrap_spearman(pairs, cluster_ids, B: int = BOOTSTRAP_B,
                               seed: int = 0) -> dict:
    """Cluster (block) bootstrap of Spearman rho with a percentile 95% CI.

    WHY: rows inside one (model, layer, task) cell are NOT independent — they
    share a single R value (see module note above). Resampling individual
    (A, R) pairs would pretend we have 270 independent observations and give a
    CI that is far too tight. Instead we resample whole clusters with
    replacement (the standard block bootstrap for clustered data), recompute
    Spearman on the pooled resample each time, and take the 2.5/97.5 percentiles.

    Args:
        pairs:       sequence of (A, R) pairs (floats), one per row.
        cluster_ids: sequence, same length as `pairs`; rows sharing an id form
                     one cluster. Default cluster in main() is (model, layer, task).
        B:           number of bootstrap resamples.
        seed:        seed for numpy.random.default_rng (determinism).

    Returns:
        {"rho": point estimate on the observed data,
         "ci_low", "ci_high": 95% percentile CI bounds,
         "B": effective number of resamples that yielded a finite rho,
         "n_clusters": number of distinct clusters}.
        rho/ci fields are nan when there is not enough variation to define them.
    """
    pairs = list(pairs)
    cluster_ids = list(cluster_ids)
    if len(pairs) != len(cluster_ids):
        raise ValueError("pairs and cluster_ids must have equal length")

    # Group row indices by cluster id.
    by_cluster: dict = defaultdict(list)
    for i, cid in enumerate(cluster_ids):
        by_cluster[cid].append(i)
    cluster_keys = list(by_cluster.keys())
    cluster_index_lists = [np.asarray(by_cluster[k]) for k in cluster_keys]
    n_clusters = len(cluster_keys)

    A_all = np.asarray([p[0] for p in pairs], dtype=float)
    R_all = np.asarray([p[1] for p in pairs], dtype=float)

    point = _spearman_rho(A_all, R_all)

    if n_clusters < 2:
        return {"rho": point, "ci_low": float("nan"),
                "ci_high": float("nan"), "B": 0, "n_clusters": n_clusters}

    rng = np.random.default_rng(seed)
    boot_rhos = []
    for _ in range(B):
        chosen = rng.integers(0, n_clusters, size=n_clusters)
        idx = np.concatenate([cluster_index_lists[c] for c in chosen])
        rho = _spearman_rho(A_all[idx], R_all[idx])
        if np.isfinite(rho):
            boot_rhos.append(rho)

    if not boot_rhos:
        return {"rho": point, "ci_low": float("nan"),
                "ci_high": float("nan"), "B": 0, "n_clusters": n_clusters}

    boot = np.asarray(boot_rhos)
    return {
        "rho": point,
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "B": int(boot.size),
        "n_clusters": n_clusters,
    }


def permutation_spearman(A, R, n_perm: int = PERMUTATION_N,
                         seed: int = 0) -> dict:
    """Permutation test for Spearman rho (two-sided p reported).

    WHY: a distribution-free null for "A and R are unrelated". We shuffle R
    against A `n_perm` times, recompute rho each time, and count how often the
    permuted |rho| meets or exceeds the observed |rho|. The +1 in numerator and
    denominator is the standard add-one correction so p is never exactly 0
    (the observed arrangement is itself a valid permutation).

    Args:
        A, R:   equal-length sequences of floats.
        n_perm: number of random permutations of R.
        seed:   seed for numpy.random.default_rng.

    Returns:
        {"rho": observed rho,
         "p": two-sided permutation p-value in (0, 1],
         "p_one_sided": one-sided (permuted rho >= observed) p-value,
         "n_perm": effective number of finite permuted rhos,
         "n": sample size}.
    """
    A = np.asarray(A, dtype=float)
    R = np.asarray(R, dtype=float)
    if A.shape != R.shape:
        raise ValueError("A and R must have equal length")
    n = A.size
    obs = _spearman_rho(A, R)
    if not np.isfinite(obs) or n < 3:
        return {"rho": obs, "p": float("nan"), "p_one_sided": float("nan"),
                "n_perm": 0, "n": int(n)}

    rng = np.random.default_rng(seed)
    ge_abs = 0   # permuted |rho| >= observed |rho|  (two-sided)
    ge_signed = 0  # permuted rho   >= observed rho    (one-sided)
    eff = 0
    R_perm = R.copy()
    for _ in range(n_perm):
        rng.shuffle(R_perm)
        rho = _spearman_rho(A, R_perm)
        if not np.isfinite(rho):
            continue
        eff += 1
        if abs(rho) >= abs(obs):
            ge_abs += 1
        if rho >= obs:
            ge_signed += 1
    two_sided = (ge_abs + 1) / (eff + 1)
    one_sided = (ge_signed + 1) / (eff + 1)
    return {
        "rho": float(obs),
        "p": float(two_sided),
        "p_one_sided": float(one_sided),
        "n_perm": eff,
        "n": int(n),
    }


def _argmax_set(values, tol: float = TIE_TOL) -> set:
    """Indices whose value is within `tol` of the maximum (the tie set).

    Values that are None (nullable R) are ignored. Returns an empty set when
    every value is None.
    """
    finite = [(i, v) for i, v in enumerate(values) if v is not None]
    if not finite:
        return set()
    m = max(v for _, v in finite)
    return {i for i, v in finite if abs(v - m) <= tol}


def tie_aware_hit_rate(cells_by_outer, tol: float = TIE_TOL) -> dict:
    """Tie-aware rank-1 agreement between arg-max-A and arg-max-R per outer cell.

    WHY: the v1 hit-rate (and predictor_eval.py:139-140) calls `max(...)`, which
    breaks ties by list order — so on the very common case where R is identical
    across archs (distinct-R == 1, see distinct_r_report) EVERY arch ties for
    max R, and list order alone decides the "winner". That inflates or deflates
    the hit rate arbitrarily. Here a hit is the OVERLAP fraction between the
    arg-max-A set and the arg-max-R set:

        hit(cell) = |argmaxA ∩ argmaxR| / |argmaxA|

    (1.0 when the unique best-A arch is among the best-R archs; fractional when
    A itself ties.) We also report how often a tie occurred on either side.

    Args:
        cells_by_outer: mapping outer_key -> list of dicts, each with keys
                        "A" and "R" (R may be None). One dict per architecture.
                        Cells with < 2 archs are skipped (nothing to rank).
        tol:            tie tolerance.

    Returns:
        {"hits": summed hit fraction (float),
         "total": number of ranked outer cells,
         "hit_rate": hits/total (0.0 if total == 0),
         "tie_fraction": fraction of ranked cells where A OR R had a tie,
         "n_skipped": cells dropped for < 2 archs or all-null R}.
    """
    hits = 0.0
    total = 0
    n_tied = 0
    n_skipped = 0
    for archs in cells_by_outer.values():
        if len(archs) < 2:
            n_skipped += 1
            continue
        a_vals = [c["A"] for c in archs]
        r_vals = [c["R"] for c in archs]
        set_a = _argmax_set(a_vals, tol)
        set_r = _argmax_set(r_vals, tol)
        if not set_a or not set_r:
            # No decodable R anywhere in this cell (or no A): can't rank.
            n_skipped += 1
            continue
        total += 1
        hits += len(set_a & set_r) / len(set_a)
        if len(set_a) > 1 or len(set_r) > 1:
            n_tied += 1
    return {
        "hits": hits,
        "total": total,
        "hit_rate": (hits / total) if total else 0.0,
        "tie_fraction": (n_tied / total) if total else 0.0,
        "n_skipped": n_skipped,
    }


def distinct_r_report(recs) -> dict:
    """Count DISTINCT R values per (model, layer, task) cell across archs×seeds.

    WHY (the headline defect): R is computed once per layer from the validation
    and intervention probes and is copied onto every arch×seed row in that cell
    (run_benchmark.run_one_cell). So within a cell the number of distinct R
    values is expected to be 1 — meaning R carries NO per-probe information and
    the whole "predict which probe is most reliable" framing has no target
    variance to predict. This report makes that visible instead of letting a
    healthy-looking Spearman hide it.

    R values are rounded before counting so floating-point noise doesn't
    masquerade as genuine variation. None (nullable R) is counted as its own
    distinct category so an all-null cell reads as distinct-R == 1, not 0.

    Reads the honest floored/nullable R from "_R_floored" when present (WS6
    preserves it before load_records overwrites "R" with the legacy value on
    v2 records); otherwise falls back to "R". Either way the defect — R being
    constant across the archs×seeds inside a cell — shows identically.

    Args:
        recs: benchmark records (dicts) with keys model, layer, task, and R
              (and optionally _R_floored).

    Returns:
        {"per_cell": {(model, layer, task): {"distinct": int, "n_rows": int,
                                             "n_null": int}},
         "histogram": {distinct_count: n_cells},   # global
         "n_cells": int,
         "frac_degenerate": fraction of cells with distinct-R == 1}.
    """
    by_cell: dict = defaultdict(list)
    for r in recs:
        rval = r["_R_floored"] if "_R_floored" in r else r.get("R")
        by_cell[(r["model"], r["layer"], r["task"])].append(rval)

    per_cell = {}
    histogram: Counter = Counter()
    n_degenerate = 0
    for cell, rvals in by_cell.items():
        seen = set()
        n_null = 0
        for v in rvals:
            if v is None:
                seen.add(None)
                n_null += 1
            else:
                seen.add(round(float(v), 9))
        distinct = len(seen)
        per_cell[cell] = {"distinct": distinct, "n_rows": len(rvals),
                          "n_null": n_null}
        histogram[distinct] += 1
        if distinct == 1:
            n_degenerate += 1

    n_cells = len(per_cell)
    return {
        "per_cell": per_cell,
        "histogram": dict(sorted(histogram.items())),
        "n_cells": n_cells,
        "frac_degenerate": (n_degenerate / n_cells) if n_cells else 0.0,
    }


def _v2_honest_stats_section(recs, cells):
    """Print the additive '== v2 honest stats ==' block.

    Kept in its own function so the exploratory main() body above is untouched
    and this section is easy to import/test in isolation. Uses the same `cells`
    (median-over-seeds per model,layer,task,arch) the exploratory code built,
    plus the raw records for the distinct-R diagnosis.
    """
    print("\n" + "=" * 70)
    print("  == v2 honest stats (WS6) ==")
    print("=" * 70)

    # ---- Distinct-R report: surface the pseudo-replication defect first. ----
    dr = distinct_r_report(recs)
    print("\n-- distinct R per (model,layer,task) cell --")
    print("  A cell with distinct-R == 1 means R NEVER depended on the probe:")
    print("  the target has no per-probe variance to predict in that cell.")
    print(f"  histogram (distinct-R -> #cells): {dr['histogram']}")
    print(f"  cells: {dr['n_cells']}   "
          f"degenerate (distinct-R==1): {dr['frac_degenerate']:.1%}")
    if dr["frac_degenerate"] >= 0.5:
        print("  ⚠ MOST cells have distinct-R == 1 — R is (near-)constant "
              "within a cell. Any A-vs-R correlation is BETWEEN cells only.")

    # ---- Cluster bootstrap of the headline Spearman. ----
    # Cluster = (model, layer, task); the arch dimension lives inside a cluster.
    pairs = []
    cluster_ids = []
    for (model, layer, task, _arch), (a, r) in cells.items():
        pairs.append((a, r))
        cluster_ids.append((model, layer, task))
    cb = cluster_bootstrap_spearman(pairs, cluster_ids)
    print("\n-- cluster bootstrap Spearman (cluster = model,layer,task) --")
    print(f"  rho={cb['rho']:.3f}  95% CI [{cb['ci_low']:.3f}, "
          f"{cb['ci_high']:.3f}]  clusters={cb['n_clusters']}  B={cb['B']}")

    # ---- Permutation test on the same median-cell pairs. ----
    A = [p[0] for p in pairs]
    R = [p[1] for p in pairs]
    perm = permutation_spearman(A, R)
    print("\n-- permutation Spearman (shuffle R) --")
    print(f"  rho={perm['rho']:.3f}  two-sided p={perm['p']:.4g}  "
          f"n={perm['n']}  perms={perm['n_perm']}")

    # ---- Tie-aware hit rate over (model, layer, task) outer cells. ----
    by_outer: dict = defaultdict(list)
    for (model, layer, task, arch), (a, r) in cells.items():
        by_outer[(model, layer, task)].append({"arch": arch, "A": a, "R": r})
    th = tie_aware_hit_rate(by_outer)
    print("\n-- tie-aware rank-1 hit rate (|argmaxA ∩ argmaxR| / |argmaxA|) --")
    print(f"  hit_rate={th['hit_rate']:.3f}  "
          f"({th['hits']:.2f}/{th['total']} cells)  "
          f"tie_fraction={th['tie_fraction']:.1%}  skipped={th['n_skipped']}")
    if th["tie_fraction"] >= 0.5:
        print("  ⚠ ties dominate: list-order max (predictor_eval.py:139-140) "
              "would be meaningless here.")


def main():
    ap = argparse.ArgumentParser(description="Exploratory benchmark statistics")
    ap.add_argument("--input-dir", default="results/benchmark",
                    help="Directory of *.jsonl benchmark records")
    args = ap.parse_args()
    bench_dir = (Path(args.input_dir) if Path(args.input_dir).is_absolute()
                 else PROJECT_ROOT / args.input_dir)
    recs = load_records(bench_dir)
    cells = cell_medians(recs)
    print(f"total probe records: {len(recs)}")
    print(f"aggregated cells (model,layer,task,arch): {len(cells)}")

    models = sorted({r["model"] for r in recs})
    tasks = sorted({r["task"] for r in recs})
    archs = sorted({r["arch"] for r in recs})

    def pairs(filt):
        return [(a, r) for k, (a, r) in cells.items() if filt(k)]

    print("\n== Per-model (270-pair subset, exploratory) ==")
    print(f"{'model':<35} {'n':>4} {'rho':>8} {'p':>10} {'med A':>8} {'med R':>8} {'hit':>7}")
    for m in models:
        ps = pairs(lambda k: k[0] == m)
        if not ps:
            continue
        rho, p = spearmanr([x[0] for x in ps], [x[1] for x in ps])
        keys = {(k[0], k[1], k[2]) for k in cells if k[0] == m}
        h, t = hit_rate(cells, keys)
        med_a = statistics.median(x[0] for x in ps)
        med_r = statistics.median(x[1] for x in ps)
        print(f"{m:<35} {len(ps):>4} {rho:>8.3f} {p:>10.2e} {med_a:>8.4f} {med_r:>8.4f} {h:>3}/{t}")

    print("\n== Per-task ==")
    for t_ in tasks:
        ps = pairs(lambda k: k[2] == t_)
        if not ps:
            continue
        rho, p = spearmanr([x[0] for x in ps], [x[1] for x in ps])
        keys = {(k[0], k[1], k[2]) for k in cells if k[2] == t_}
        h, tt = hit_rate(cells, keys)
        print(f"{t_:<10} n={len(ps):>3}  rho={rho:>7.3f}  p={p:.2e}  hit={h}/{tt}")

    print("\n== Per-architecture (within-arch correlation across cells) ==")
    for a_ in archs:
        ps = pairs(lambda k: k[3] == a_)
        accs = [r["accuracy"] for r in recs if r["arch"] == a_]
        if not ps or not accs:
            continue
        rho, p = spearmanr([x[0] for x in ps], [x[1] for x in ps])
        med_a = statistics.median(x[0] for x in ps)
        med_r = statistics.median(x[1] for x in ps)
        print(f"{a_:<8} n={len(ps):>3}  rho={rho:>7.3f}  p={p:.2e}  "
              f"med A={med_a:.4f}  med R={med_r:.4f}  med acc={statistics.median(accs):.3f}")

    print("\n== Per-layer-position (1=shallowest of the 5 sampled) ==")
    layer_rank = {}
    for m in models:
        ls = sorted({r["layer"] for r in recs if r["model"] == m})
        for i, l in enumerate(ls):
            layer_rank[(m, l)] = i + 1
    # Iterate over the layer positions actually present rather than a hardcoded
    # range(1, 6): a partial run, a single model, or any model whose
    # select_layers returned fewer than 5 layers leaves higher positions empty,
    # and statistics.median on the empty bucket raised StatisticsError — which
    # aborted main() BEFORE the WS6 v2-honest-stats section below could run.
    # In the intended full 5-layer run the present positions are exactly
    # {1..5}, so this loop's output is unchanged.
    for pos in sorted(set(layer_rank.values())):
        ps = pairs(lambda k: layer_rank.get((k[0], k[1])) == pos)
        if not ps:
            continue
        rho, p = spearmanr([x[0] for x in ps], [x[1] for x in ps])
        med_r = statistics.median(x[1] for x in ps)
        print(f"pos {pos}: n={len(ps):>3}  rho={rho:>7.3f}  p={p:.2e}  med R={med_r:.4f}")

    print("\n== Winning intervention method (per-probe R_method) ==")
    meth = defaultdict(int)
    for r in recs:
        meth[r["R_method"]] += 1
    for k, v in sorted(meth.items(), key=lambda kv: -kv[1]):
        print(f"{k:<10} {v:>5}  ({v/len(recs):.1%})")

    print("\n== Probe accuracy by task (sanity: probes are above chance) ==")
    for t_ in tasks:
        accs = [r["accuracy"] for r in recs if r["task"] == t_]
        print(f"{t_:<10} median={statistics.median(accs):.3f}  "
              f"min={min(accs):.3f}  max={max(accs):.3f}")

    print("\n== R distribution ==")
    rs = sorted(r["R"] for r in recs)
    n = len(rs)
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        print(f"q{int(q*100):02d}: {rs[int(q*(n-1))]:.4f}")
    print(f"frac R<0.5: {sum(r < 0.5 for r in rs)/n:.1%}")

    print("\n== lambda_max baseline (pre-specified exploratory) ==")
    groups = defaultdict(lambda: {"L": [], "R": []})
    for r in recs:
        key = (r["model"], r["layer"], r["task"], r["arch"])
        groups[key]["L"].append(r["lambda_max"])
        groups[key]["R"].append(r["R"])
    lam = [(statistics.median(v["L"]), statistics.median(v["R"])) for v in groups.values()]
    rho, p = spearmanr([x[0] for x in lam], [x[1] for x in lam])
    print(f"lambda_max vs R (270 cells): rho={rho:.3f}  p={p:.2e}")

    total_s = sum(r["wallclock_s"] for r in recs)
    print(f"\ntotal probe-training+eval wallclock: {total_s/3600:.1f} h "
          f"(sum of per-record wallclock_s)")

    # WS6 additive section. Appended AFTER all exploratory output so nothing
    # above changes. Uses the raw `recs` for the distinct-R diagnosis (which
    # needs per-seed rows), not the median-collapsed `cells`.
    _v2_honest_stats_section(recs, cells)


if __name__ == "__main__":
    main()
