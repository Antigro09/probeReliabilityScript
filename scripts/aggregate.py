"""
Aggregate per-model results into the cross-model comparison table.

Usage:
    python -m scripts.aggregate

Reads results/*/results.json and emits:
    results/aggregate/summary.csv             Long-format per (model, layer, probe, intervention)
    results/aggregate/best_per_model.csv      One row per model: best layer, best R, MKA, lambda_max
    results/aggregate/main_table.csv          The Table 1 in the paper

Schema of summary.csv:
    model, layer, probe, intervention, C, S, R, accuracy,
    mka_alignment, lambda_max, mean_align_top
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    results_dir = PROJECT_ROOT / "results"
    if not results_dir.exists():
        print(f"❌ No results directory at {results_dir}")
        return rows

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if model_dir.name == "aggregate":
            continue
        results_json = model_dir / "results.json"
        if not results_json.exists():
            continue
        with results_json.open() as f:
            data = json.load(f)
        model_name = data["model"]
        for layer_data in data["results"]:
            layer = layer_data["layer"]
            accs = layer_data["accuracies"]
            mka_align = layer_data["mka_alignment"]
            hess = layer_data["hessian"]
            interventions = layer_data["interventions"]
            for probe_name in ["linear", "mlp", "mka"]:
                acc = accs.get(probe_name, float("nan"))
                mka_a = mka_align.get(probe_name, float("nan"))
                lam_max = hess.get(probe_name, {}).get("lambda_max", float("nan"))
                m_align = hess.get(probe_name, {}).get("mean_align_top", float("nan"))
                for method, m in interventions.get(probe_name, {}).items():
                    rows.append({
                        "model": model_name,
                        "layer": layer,
                        "probe": probe_name,
                        "intervention": method,
                        "C": m["C"], "S": m["S"], "R": m["R"],
                        "accuracy": acc,
                        "mka_alignment": mka_a,
                        "lambda_max": lam_max,
                        "mean_align_top": m_align,
                    })
    return rows


def write_summary(rows: list[dict], out_path: Path) -> None:
    import csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("⚠ No rows to write")
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def make_main_table(rows: list[dict]) -> list[dict]:
    """
    For each (model, layer, probe), pick the BEST intervention by R.
    Then for each model, report the overall best (layer, R, MKA, lambda_max).

    This reproduces Table 1 in the paper.
    """
    if not rows:
        return []
    best_per_model: dict[str, dict] = {}
    for r in rows:
        key = r["model"]
        if key not in best_per_model or r["R"] > best_per_model[key]["R"]:
            best_per_model[key] = dict(r)  # copy
    return list(best_per_model.values())


def main():
    print("Aggregating results...")
    rows = collect_rows()
    print(f"  Collected {len(rows)} intervention rows across "
          f"{len(set(r['model'] for r in rows))} models")
    out_dir = PROJECT_ROOT / "results" / "aggregate"
    write_summary(rows, out_dir / "summary.csv")
    main_table = make_main_table(rows)
    write_summary(main_table, out_dir / "main_table.csv")
    print(f"  Wrote {out_dir / 'summary.csv'}")
    print(f"  Wrote {out_dir / 'main_table.csv'}")
    print()
    print("Main table preview:")
    for r in main_table:
        print(f"  {r['model']:50s}  L{r['layer']:>2d}  "
              f"{r['probe']:7s}  {r['intervention']:8s}  "
              f"R={r['R']:.3f}  MKA={r['mka_alignment']:.3f}  "
              f"λmax={r['lambda_max']:.1f}")


if __name__ == "__main__":
    main()
