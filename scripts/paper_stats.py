"""Exploratory descriptive statistics for the paper write-up.

NOT part of the pre-registered analysis (that is scripts/predictor_eval.py).
This script aggregates the same 5,400 benchmark records into per-model,
per-task, per-architecture, and per-layer-depth tables for reporting.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr

BENCH = Path(__file__).resolve().parent.parent / "results" / "benchmark"


def load_records():
    recs = []
    for f in sorted(BENCH.glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                recs.append(json.loads(line))
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


def main():
    recs = load_records()
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
        rho, p = spearmanr([x[0] for x in ps], [x[1] for x in ps])
        keys = {(k[0], k[1], k[2]) for k in cells if k[0] == m}
        h, t = hit_rate(cells, keys)
        med_a = statistics.median(x[0] for x in ps)
        med_r = statistics.median(x[1] for x in ps)
        print(f"{m:<35} {len(ps):>4} {rho:>8.3f} {p:>10.2e} {med_a:>8.4f} {med_r:>8.4f} {h:>3}/{t}")

    print("\n== Per-task ==")
    for t_ in tasks:
        ps = pairs(lambda k: k[2] == t_)
        rho, p = spearmanr([x[0] for x in ps], [x[1] for x in ps])
        keys = {(k[0], k[1], k[2]) for k in cells if k[2] == t_}
        h, tt = hit_rate(cells, keys)
        print(f"{t_:<10} n={len(ps):>3}  rho={rho:>7.3f}  p={p:.2e}  hit={h}/{tt}")

    print("\n== Per-architecture (within-arch correlation across cells) ==")
    for a_ in archs:
        ps = pairs(lambda k: k[3] == a_)
        rho, p = spearmanr([x[0] for x in ps], [x[1] for x in ps])
        med_a = statistics.median(x[0] for x in ps)
        med_r = statistics.median(x[1] for x in ps)
        accs = [r["accuracy"] for r in recs if r["arch"] == a_]
        print(f"{a_:<8} n={len(ps):>3}  rho={rho:>7.3f}  p={p:.2e}  "
              f"med A={med_a:.4f}  med R={med_r:.4f}  med acc={statistics.median(accs):.3f}")

    print("\n== Per-layer-position (1=shallowest of the 5 sampled) ==")
    layer_rank = {}
    for m in models:
        ls = sorted({r["layer"] for r in recs if r["model"] == m})
        for i, l in enumerate(ls):
            layer_rank[(m, l)] = i + 1
    for pos in range(1, 6):
        ps = pairs(lambda k: layer_rank.get((k[0], k[1])) == pos)
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


if __name__ == "__main__":
    main()
