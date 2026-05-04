"""
Pre-registered predictor evaluation.

Reads results/benchmark/*.jsonl from run_benchmark.py and tests the three
pre-registered predictions against the data:

    P1 (primary): Spearman rho >= 0.5 with p < 0.01 across all
                  (alignment, reliability) pairs.

    P2 (secondary): Architecture with highest median A also has highest
                    median R in >= 50% of (model, layer, task) cells.

    P3 (robustness): P1 (rho >= 0.4) and P2 (>= 40%) hold within EACH task.

Output: results/benchmark/PREREG_OUTCOME.json plus a console summary.

This script is locked at the pre-registration time. Do not modify it after
the benchmark has been run -- doing so violates the pre-registration.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy.stats import spearmanr


# ---- Locked thresholds (pre-registered) ----
P1_RHO_THRESHOLD = 0.5
P1_P_THRESHOLD = 0.01
P2_HITRATE_THRESHOLD = 0.50
P3_TASK_RHO_THRESHOLD = 0.4
P3_TASK_HITRATE_THRESHOLD = 0.40


def load_benchmark_rows() -> list[dict]:
    """Load every line of every results/benchmark/*.jsonl file."""
    bench_dir = PROJECT_ROOT / "results" / "benchmark"
    if not bench_dir.exists():
        return []
    rows: list[dict] = []
    for f in sorted(bench_dir.glob("*.jsonl")):
        with f.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def aggregate_by_cell(rows: list[dict]) -> dict[tuple, dict]:
    """
    Group rows by (model, layer, task, arch) cell.
    Returns dict[cell_key] -> {"A_median": ..., "R_median": ...,
                                "A_seeds": [...], "R_seeds": [...], "n": K}
    """
    grouped: dict[tuple, dict] = defaultdict(
        lambda: {"A_seeds": [], "R_seeds": [], "acc_seeds": []}
    )
    for r in rows:
        key = (r["model"], r["layer"], r["task"], r["arch"])
        grouped[key]["A_seeds"].append(r["A"])
        grouped[key]["R_seeds"].append(r["R"])
        grouped[key]["acc_seeds"].append(r["accuracy"])

    out: dict[tuple, dict] = {}
    for key, d in grouped.items():
        out[key] = {
            "model": key[0], "layer": key[1], "task": key[2], "arch": key[3],
            "A_median": float(np.median(d["A_seeds"])),
            "R_median": float(np.median(d["R_seeds"])),
            "acc_median": float(np.median(d["acc_seeds"])),
            "A_seeds": d["A_seeds"],
            "R_seeds": d["R_seeds"],
            "n": len(d["A_seeds"]),
        }
    return out


def evaluate_p1(cell_aggs: dict[tuple, dict]) -> dict:
    """Spearman rho between A_median and R_median across all cells."""
    A = [c["A_median"] for c in cell_aggs.values()]
    R = [c["R_median"] for c in cell_aggs.values()]
    if len(A) < 3:
        return {"met": False, "reason": "Too few cells", "n": len(A)}
    rho, p = spearmanr(A, R)
    return {
        "rho": float(rho), "p": float(p), "n": len(A),
        "threshold_rho": P1_RHO_THRESHOLD,
        "threshold_p": P1_P_THRESHOLD,
        "met": (rho >= P1_RHO_THRESHOLD) and (p < P1_P_THRESHOLD),
    }


def evaluate_p2(cell_aggs: dict[tuple, dict]) -> dict:
    """
    For each (model, layer, task) outer cell, look at the 3 architectures.
    Does the architecture with highest A_median also have highest R_median?
    """
    by_outer: dict[tuple, list[dict]] = defaultdict(list)
    for cell in cell_aggs.values():
        outer = (cell["model"], cell["layer"], cell["task"])
        by_outer[outer].append(cell)

    n_total = 0
    n_hit = 0
    for outer, cells in by_outer.items():
        if len(cells) < 2:
            continue   # need at least 2 architectures to rank
        n_total += 1
        # Architecture with highest A
        best_A_arch = max(cells, key=lambda c: c["A_median"])["arch"]
        best_R_arch = max(cells, key=lambda c: c["R_median"])["arch"]
        if best_A_arch == best_R_arch:
            n_hit += 1
    hit_rate = n_hit / n_total if n_total > 0 else 0.0
    return {
        "hit_rate": hit_rate, "n_hit": n_hit, "n_total": n_total,
        "threshold": P2_HITRATE_THRESHOLD,
        "met": hit_rate >= P2_HITRATE_THRESHOLD,
    }


def evaluate_p3(cell_aggs: dict[tuple, dict]) -> dict:
    """P1 and P2 both hold per-task with relaxed thresholds."""
    by_task: dict[str, dict] = defaultdict(dict)
    for k, c in cell_aggs.items():
        task = c["task"]
        by_task[task][k] = c

    per_task = {}
    all_met = True
    for task, task_cells in by_task.items():
        # Per-task Spearman
        A = [c["A_median"] for c in task_cells.values()]
        R = [c["R_median"] for c in task_cells.values()]
        if len(A) >= 3:
            rho, p = spearmanr(A, R)
            rho_met = rho >= P3_TASK_RHO_THRESHOLD
        else:
            rho, p, rho_met = float("nan"), float("nan"), False

        # Per-task hit rate
        by_outer: dict[tuple, list[dict]] = defaultdict(list)
        for c in task_cells.values():
            by_outer[(c["model"], c["layer"])].append(c)
        n_total = 0
        n_hit = 0
        for cells in by_outer.values():
            if len(cells) < 2:
                continue
            n_total += 1
            best_A = max(cells, key=lambda x: x["A_median"])["arch"]
            best_R = max(cells, key=lambda x: x["R_median"])["arch"]
            if best_A == best_R:
                n_hit += 1
        hr = n_hit / n_total if n_total > 0 else 0.0
        hr_met = hr >= P3_TASK_HITRATE_THRESHOLD

        task_met = rho_met and hr_met
        if not task_met:
            all_met = False
        per_task[task] = {
            "rho": float(rho), "p": float(p),
            "hit_rate": hr, "n_hit": n_hit, "n_total": n_total,
            "rho_met": rho_met, "hr_met": hr_met, "met": task_met,
        }

    return {
        "per_task": per_task,
        "threshold_rho": P3_TASK_RHO_THRESHOLD,
        "threshold_hit_rate": P3_TASK_HITRATE_THRESHOLD,
        "met": all_met and len(per_task) > 0,
    }


def classify_outcome(p1: dict, p2: dict, p3: dict) -> str:
    """Maps to the outcome rubric in PREREGISTRATION.md."""
    if p1["met"] and p2["met"] and p3["met"]:
        return "STRONG_POSITIVE"
    if p1["met"] and p2["met"] and not p3["met"]:
        return "AGGREGATE_POSITIVE_NO_GENERALIZATION"
    if p1["met"] and not p2["met"]:
        return "OBSERVATIONAL_NOT_OPERATIONAL"
    return "NEGATIVE"


def main():
    rows = load_benchmark_rows()
    if not rows:
        print("❌ No benchmark results found in results/benchmark/")
        print("   Run scripts/run_benchmark.py first.")
        sys.exit(1)
    print(f"Loaded {len(rows)} probe records")

    cell_aggs = aggregate_by_cell(rows)
    print(f"Aggregated into {len(cell_aggs)} cells "
          f"(model x layer x task x arch)")

    # ---- Pre-registered tests ----
    p1 = evaluate_p1(cell_aggs)
    p2 = evaluate_p2(cell_aggs)
    p3 = evaluate_p3(cell_aggs)
    outcome = classify_outcome(p1, p2, p3)

    # ---- Print summary ----
    print("\n" + "=" * 70)
    print("  PRE-REGISTERED PREDICTIONS")
    print("=" * 70)
    print(f"\nP1 (primary): Spearman rho >= {P1_RHO_THRESHOLD}, p < {P1_P_THRESHOLD}")
    print(f"   rho = {p1.get('rho', 'NA'):.4f}  "
          f"p = {p1.get('p', 'NA'):.4g}  n = {p1.get('n', 'NA')}")
    print(f"   {'✅ MET' if p1['met'] else '❌ NOT MET'}")

    print(f"\nP2 (secondary): rank-1 hit rate >= {P2_HITRATE_THRESHOLD:.0%}")
    print(f"   hit rate = {p2.get('hit_rate', 0):.3f} "
          f"({p2.get('n_hit', 0)}/{p2.get('n_total', 0)} cells)")
    print(f"   {'✅ MET' if p2['met'] else '❌ NOT MET'}")

    print(f"\nP3 (robustness): per-task rho >= {P3_TASK_RHO_THRESHOLD}, "
          f"per-task hit rate >= {P3_TASK_HITRATE_THRESHOLD:.0%}")
    for task, t in p3.get("per_task", {}).items():
        flag = "✅" if t["met"] else "❌"
        print(f"   {flag} {task:8s}  rho={t['rho']:.3f}  "
              f"hit={t['hit_rate']:.3f}  ({t['n_hit']}/{t['n_total']})")
    print(f"   {'✅ ALL TASKS MET' if p3['met'] else '❌ NOT ALL TASKS MET'}")

    print("\n" + "=" * 70)
    print(f"  OUTCOME: {outcome}")
    print("=" * 70)

    # ---- Save full result for the paper ----
    def _native(obj):
        """Recursively convert numpy scalars/bools to native Python types."""
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

    out = {
        "thresholds": {
            "P1_rho": P1_RHO_THRESHOLD, "P1_p": P1_P_THRESHOLD,
            "P2_hit_rate": P2_HITRATE_THRESHOLD,
            "P3_rho": P3_TASK_RHO_THRESHOLD,
            "P3_hit_rate": P3_TASK_HITRATE_THRESHOLD,
        },
        "P1": p1, "P2": p2, "P3": p3,
        "outcome": outcome,
        "n_probes_total": len(rows),
        "n_cells": len(cell_aggs),
    }
    out = _native(out)
    out_path = PROJECT_ROOT / "results" / "benchmark" / "PREREG_OUTCOME.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
