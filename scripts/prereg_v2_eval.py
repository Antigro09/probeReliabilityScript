"""
Pre-registered predictor evaluation, v2 (locked).

WHY a second evaluator: predictor_eval.py is locked to the v1
pre-registration and REFUSES v2 records (schema_version >= 2). v2 changed the
target: reliability R is now FLOORED and NULLABLE (src/metrics.py) — a cell
whose validation probe cannot decode the feature from clean representations
yields R = None instead of the v1 constant-1.0 pathology. Correlating against a
nullable target, and honestly accounting for the fact that R is shared across
the archs×seeds inside a cell, needs its own locked analysis. This is it.

What this script does differently from v1:

  * REQUIRES schema_version == 2 (v1 refuses >= 2; this one refuses != 2), so
    v1 and v2 records can never blend invisibly into one analysis.

  * Uses the FLOORED, nullable R (record key "R"), NOT the legacy clipped
    R_legacy. Cells whose median R is None are excluded from the correlations
    and the excluded count (n_null) is surfaced everywhere.

  * Runs a DISTINCT-R SANITY GATE first. Because R is computed once per
    (model, layer, task) cell and copied onto every arch×seed row, if most
    cells have only one distinct R value then the target has no per-probe
    variance and the "which architecture is most reliable?" question is
    vacuous. In that regime the evaluation ABORTS rather than reporting a
    between-cell correlation dressed up as an operational result.

  * Reports the cluster-bootstrap CI and the tie-aware hit rate from
    scripts/paper_stats (shared, tested there), not the tie-blind list-order
    max that predictor_eval.py:139-140 uses.

Output: <input-dir>/PREREG_V2_OUTCOME.json plus a console summary.

This script is LOCKED at v2 pre-registration time. Its thresholds and
statistics must not change after the benchmark has been run — doing so
violates the pre-registration.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy.stats import spearmanr

# Shared, unit-tested honest-stats helpers (WS6). Importing them here keeps the
# bootstrap / tie-aware / distinct-R logic in ONE place with ONE test suite.
from scripts.paper_stats import (
    cluster_bootstrap_spearman,
    tie_aware_hit_rate,
    distinct_r_report,
    permutation_spearman,
)

SCHEMA_VERSION_REQUIRED = 2


# ==========================================================================
# LOCKED THRESHOLDS (pre-registered v2).
#
# Every value below is an experimenter choice. They are PLACEHOLDERS until the
# pre-registration document fixes them; do not treat a run made against these
# defaults as the locked result.
#     -> lock in PREREGISTRATION_v2.md
# ==========================================================================
# P1 (primary): Spearman rho between median A and median (floored) R across
# non-null cells must reach P1_RHO with permutation/asymptotic p < P1_P.
P1_RHO = 0.5                 # lock in PREREGISTRATION_v2.md
P1_P = 0.01                  # lock in PREREGISTRATION_v2.md
# P2 (secondary): tie-aware rank-1 hit rate must reach this.
P2_HITRATE = 0.50            # lock in PREREGISTRATION_v2.md
# Distinct-R sanity gate: if the fraction of (model,layer,task) cells whose
# floored R takes only ONE distinct value across archs×seeds is at or above
# this, the target has no per-probe variance and the eval ABORTS. This is the
# central v2 guard — without per-cell R variance, any A-vs-R correlation is a
# between-cell artifact, not evidence that A predicts probe reliability.
DISTINCT_R_MIN = 0.50        # lock in PREREGISTRATION_v2.md
# Resample budgets for the reported CI / permutation p. Analysis-power choices.
BOOTSTRAP_B = 2000           # lock in PREREGISTRATION_v2.md
PERMUTATION_N = 10000        # lock in PREREGISTRATION_v2.md


def load_v2_rows(bench_dir: Path) -> list[dict]:
    """Load every *.jsonl line in `bench_dir`, requiring schema_version == 2.

    Refuses any row that is not exactly v2. v1 rows and any future schema must
    be evaluated by their own locked evaluator; mixing is how tainted and clean
    records blend invisibly.
    """
    if not bench_dir.exists():
        return []
    rows: list[dict] = []
    for f in sorted(bench_dir.glob("*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                ver = row.get("schema_version", 1)
                if ver != SCHEMA_VERSION_REQUIRED:
                    sys.exit(
                        f"{f}:{lineno}: row has schema_version={ver}. This "
                        f"evaluator is locked to PREREGISTRATION_v2 and "
                        f"requires schema_version == {SCHEMA_VERSION_REQUIRED}. "
                        f"Use predictor_eval.py for v1 records; keep versions "
                        f"in separate directories."
                    )
                rows.append(row)
    return rows


def aggregate_by_cell(rows: list[dict]) -> dict[tuple, dict]:
    """Median-over-seeds per (model, layer, task, arch) cell.

    R uses the FLOORED nullable target (record key "R"). Because R is shared
    across seeds in a cell it is typically constant, but we still take the
    median of the non-null seed values so a partially-null cell degrades
    gracefully. A cell whose R is null for EVERY seed has R_median = None and
    is dropped from the correlations downstream (with the count surfaced).
    """
    grouped: dict[tuple, dict] = defaultdict(
        lambda: {"A": [], "R_nonnull": [], "acc": [], "n_rows": 0, "n_null": 0}
    )
    for r in rows:
        key = (r["model"], r["layer"], r["task"], r["arch"])
        g = grouped[key]
        g["A"].append(r["A"])
        g["acc"].append(r["accuracy"])
        g["n_rows"] += 1
        if r["R"] is None:
            g["n_null"] += 1
        else:
            g["R_nonnull"].append(r["R"])

    out: dict[tuple, dict] = {}
    for key, g in grouped.items():
        r_median = (statistics.median(g["R_nonnull"])
                    if g["R_nonnull"] else None)
        out[key] = {
            "model": key[0], "layer": key[1], "task": key[2], "arch": key[3],
            "A_median": float(statistics.median(g["A"])),
            "R_median": (float(r_median) if r_median is not None else None),
            "acc_median": float(statistics.median(g["acc"])),
            "n_rows": g["n_rows"],
            "n_null_rows": g["n_null"],
        }
    return out


def run_distinct_r_gate(rows: list[dict]) -> dict:
    """Distinct-R sanity gate on the raw (per-seed, floored) records.

    Returns the distinct_r_report plus a pass/fail against DISTINCT_R_MIN.
    Gate FAILS (aborts the eval) when frac_degenerate >= DISTINCT_R_MIN.
    """
    dr = distinct_r_report(rows)
    passed = dr["frac_degenerate"] < DISTINCT_R_MIN
    return {
        "histogram": dr["histogram"],
        "n_cells": dr["n_cells"],
        "frac_degenerate": dr["frac_degenerate"],
        "threshold": DISTINCT_R_MIN,
        "passed": passed,
    }


def evaluate_p1(cell_aggs: dict[tuple, dict]) -> dict:
    """Spearman rho between median A and median (floored) R over NON-NULL cells.

    Null-R cells are excluded and counted (n_null). The p-value is the
    permutation two-sided p (distribution-free); the asymptotic p is reported
    alongside for reference. The cluster bootstrap CI (cluster = model,layer,
    task) is reported so the pseudo-replication from shared-R is visible.
    """
    used = [c for c in cell_aggs.values() if c["R_median"] is not None]
    n_null = sum(1 for c in cell_aggs.values() if c["R_median"] is None)
    if len(used) < 3:
        return {"met": False, "reason": "Too few non-null cells",
                "n": len(used), "n_null": n_null}

    A = [c["A_median"] for c in used]
    R = [c["R_median"] for c in used]
    rho_asym, p_asym = spearmanr(A, R)

    perm = permutation_spearman(A, R, n_perm=PERMUTATION_N, seed=0)
    pairs = list(zip(A, R))
    cluster_ids = [(c["model"], c["layer"], c["task"]) for c in used]
    cb = cluster_bootstrap_spearman(pairs, cluster_ids, B=BOOTSTRAP_B, seed=0)

    rho = float(rho_asym)
    p = float(perm["p"])   # permutation p is the pre-registered decision p
    return {
        "rho": rho,
        "p": p,
        "p_asymptotic": float(p_asym),
        "n": len(used),
        "n_null": n_null,
        "cluster_bootstrap": cb,
        "threshold_rho": P1_RHO,
        "threshold_p": P1_P,
        "met": (rho >= P1_RHO) and (p < P1_P),
    }


def evaluate_p2(cell_aggs: dict[tuple, dict]) -> dict:
    """Tie-aware rank-1 hit rate over (model, layer, task) outer cells.

    Uses the shared tie_aware_hit_rate: a hit is |argmaxA ∩ argmaxR|/|argmaxA|,
    not the list-order max. Null R values inside a cell are ignored by the
    helper; a cell with no decodable R anywhere is skipped.
    """
    by_outer: dict[tuple, list[dict]] = defaultdict(list)
    for c in cell_aggs.values():
        by_outer[(c["model"], c["layer"], c["task"])].append(
            {"arch": c["arch"], "A": c["A_median"], "R": c["R_median"]}
        )
    th = tie_aware_hit_rate(by_outer)
    return {
        "hit_rate": th["hit_rate"],
        "hits": th["hits"],
        "total": th["total"],
        "tie_fraction": th["tie_fraction"],
        "n_skipped": th["n_skipped"],
        "threshold": P2_HITRATE,
        "met": th["hit_rate"] >= P2_HITRATE,
    }


def classify_outcome(gate: dict, p1: dict, p2: dict) -> str:
    """Map to the v2 outcome rubric (mirror in PREREGISTRATION_v2.md)."""
    if not gate["passed"]:
        return "ABORTED_NO_TARGET_VARIANCE"
    if p1["met"] and p2["met"]:
        return "STRONG_POSITIVE"
    if p1["met"] and not p2["met"]:
        return "OBSERVATIONAL_NOT_OPERATIONAL"
    return "NEGATIVE"


def _native(obj):
    """Recursively convert numpy scalars/bools to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: _native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_native(x) for x in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate the v2 pre-registered predictions "
                    "(floored/nullable R) on schema-v2 benchmark records"
    )
    ap.add_argument("--input-dir", default="results/benchmark_v2",
                    help="Directory of *.jsonl benchmark records (v2 schema)")
    args = ap.parse_args()
    bench_dir = (Path(args.input_dir) if Path(args.input_dir).is_absolute()
                 else PROJECT_ROOT / args.input_dir)

    rows = load_v2_rows(bench_dir)
    if not rows:
        print(f"❌ No v2 benchmark results found in {bench_dir}")
        print("   Run scripts/run_benchmark.py (writes schema v2) first.")
        sys.exit(1)
    print(f"Loaded {len(rows)} v2 probe records from {bench_dir}")

    cell_aggs = aggregate_by_cell(rows)
    n_null_cells = sum(1 for c in cell_aggs.values() if c["R_median"] is None)
    print(f"Aggregated into {len(cell_aggs)} cells "
          f"(model x layer x task x arch); {n_null_cells} have null median R")

    # ---- Distinct-R sanity gate FIRST (may abort). ----
    gate = run_distinct_r_gate(rows)
    print("\n" + "=" * 70)
    print("  DISTINCT-R SANITY GATE")
    print("=" * 70)
    print(f"  histogram (distinct-R per cell -> #cells): {gate['histogram']}")
    print(f"  degenerate (distinct-R==1): {gate['frac_degenerate']:.1%}  "
          f"(gate aborts at >= {gate['threshold']:.0%})")
    print(f"  {'✅ PASS' if gate['passed'] else '❌ FAIL — no per-probe R variance'}")

    if not gate["passed"]:
        outcome = classify_outcome(gate, {"met": False}, {"met": False})
        print("\n" + "=" * 70)
        print(f"  OUTCOME: {outcome}")
        print("=" * 70)
        print("  R is (near-)constant within cells, so A cannot be shown to "
              "predict\n  PER-PROBE reliability. Any A-vs-R correlation would "
              "be a between-cell\n  artifact. Refusing to report P1/P2.")
        out = _native({
            "schema_version_required": SCHEMA_VERSION_REQUIRED,
            "thresholds": {
                "P1_rho": P1_RHO, "P1_p": P1_P, "P2_hit_rate": P2_HITRATE,
                "distinct_r_min": DISTINCT_R_MIN,
                "bootstrap_B": BOOTSTRAP_B, "permutation_n": PERMUTATION_N,
            },
            "distinct_r_gate": gate,
            "outcome": outcome,
            "n_probes_total": len(rows),
            "n_cells": len(cell_aggs),
            "n_null_cells": n_null_cells,
        })
        out_path = bench_dir / "PREREG_V2_OUTCOME.json"
        out_path.write_text(json.dumps(out, indent=2))
        print(f"\nSaved: {out_path}")
        sys.exit(3)

    # ---- Pre-registered tests. ----
    p1 = evaluate_p1(cell_aggs)
    p2 = evaluate_p2(cell_aggs)
    outcome = classify_outcome(gate, p1, p2)

    print("\n" + "=" * 70)
    print("  PRE-REGISTERED PREDICTIONS (v2, floored/nullable R)")
    print("=" * 70)

    print(f"\nP1 (primary): Spearman rho >= {P1_RHO}, permutation p < {P1_P}")
    if "rho" in p1:
        cb = p1["cluster_bootstrap"]
        print(f"   rho = {p1['rho']:.4f}  perm p = {p1['p']:.4g}  "
              f"(asymptotic p = {p1['p_asymptotic']:.4g})")
        print(f"   cluster-bootstrap 95% CI [{cb['ci_low']:.3f}, "
              f"{cb['ci_high']:.3f}]  clusters={cb['n_clusters']}")
        print(f"   n = {p1['n']} non-null cells  "
              f"(excluded {p1['n_null']} null-R cells)")
    else:
        print(f"   {p1.get('reason', 'n/a')}  n = {p1.get('n', 0)}")
    print(f"   {'✅ MET' if p1['met'] else '❌ NOT MET'}")

    print(f"\nP2 (secondary): tie-aware rank-1 hit rate >= {P2_HITRATE:.0%}")
    print(f"   hit rate = {p2['hit_rate']:.3f} "
          f"({p2['hits']:.2f}/{p2['total']} cells)  "
          f"tie_fraction = {p2['tie_fraction']:.1%}  "
          f"skipped = {p2['n_skipped']}")
    print(f"   {'✅ MET' if p2['met'] else '❌ NOT MET'}")

    print("\n" + "=" * 70)
    print(f"  OUTCOME: {outcome}")
    print("=" * 70)

    out = _native({
        "schema_version_required": SCHEMA_VERSION_REQUIRED,
        "thresholds": {
            "P1_rho": P1_RHO, "P1_p": P1_P, "P2_hit_rate": P2_HITRATE,
            "distinct_r_min": DISTINCT_R_MIN,
            "bootstrap_B": BOOTSTRAP_B, "permutation_n": PERMUTATION_N,
        },
        "distinct_r_gate": gate,
        "P1": p1, "P2": p2,
        "outcome": outcome,
        "n_probes_total": len(rows),
        "n_cells": len(cell_aggs),
        "n_null_cells": n_null_cells,
    })
    out_path = bench_dir / "PREREG_V2_OUTCOME.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
