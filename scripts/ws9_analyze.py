"""Analyze schema-v3 robustness records and create paper-ready artifacts.

The analysis keeps fixed-evaluator damage separate from post-edit
redecodability.  It reports model-task block-bootstrap uncertainty,
leave-one-model/task-out sensitivity, Hessian-subspace ``k`` sensitivity,
all requested layer exclusions, matched baselines, and perturbation sweeps.
Figures use the manuscript's gray/salmon/crimson palette.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.paper_stats import cluster_bootstrap_spearman
from src.robustness import orientation_robust_cs

SCHEMA_VERSION = 3
GRAY = "#808080"
SALMON = "#E8A0A0"
CRIMSON = "#D1495B"
TREND = "#D62728"
ARCH_COLOR = {"linear": GRAY, "mlp": SALMON, "mka": CRIMSON}
ARCH_MARKER = {"linear": "o", "mlp": "s", "mka": "^"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="results/robustness_v3")
    p.add_argument("--out-dir", default=None,
                   help="default: <input-dir>/analysis")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--auc-floors", default="0.55,0.60",
                   help="clean-evaluator AUC floors for sensitivity analysis")
    return p.parse_args()


def _load_jsonl_files(paths):
    rows = []
    for path in sorted(paths):
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("schema_version") != SCHEMA_VERSION:
                    raise SystemExit(
                        f"{path}:{lineno}: expected schema {SCHEMA_VERSION}, "
                        f"got {row.get('schema_version')}"
                    )
                rows.append(row)
    return rows


def _nested(row, *keys):
    cur = row
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _finite(value):
    return value is not None and math.isfinite(float(value))


def _robust_erasure_from_auc(auc):
    if not _finite(auc):
        return None
    return max(0.0, min(1.0, 1.0 - 2.0 * (float(auc) - 0.5)))


def _floor_column(floor):
    return f"R_fixed_auc_floor_{float(floor):.2f}".replace(".", "p")


def candidate_table(rows, auc_floors):
    out = []
    all_ks = sorted({
        int(k) for row in rows for k in row.get("hessian", {}).get("A_subspace", {})
    })
    for row in rows:
        candidate = row.get("edits", {}).get("candidate_r1", {})
        evaluation = candidate.get("evaluation") or {}
        fixed = _nested(evaluation, "fixed", "aggregate") or {}
        retrained = _nested(evaluation, "retrained_summary") or {}
        target_auc = _nested(retrained, "target", "max_oriented_roc_auc")
        control_auc = _nested(retrained, "control", "max_oriented_roc_auc")
        item = {
            "model": row["model"],
            "task": row["task"],
            "position": row.get("position", "last"),
            "layer": row["layer"],
            "arch": row["arch"],
            "seed": row["seed"],
            "deep_evaluated": bool(row.get("deep_evaluated")),
            "candidate_accuracy": row.get("candidate_accuracy"),
            "A_raw": _nested(row, "hessian", "A_raw_mean_abs_cosine"),
            "R_fixed_auc": fixed.get("R_auc_oriented"),
            "C_fixed_auc": fixed.get("C_auc_oriented"),
            "S_fixed_auc": fixed.get("S_auc_oriented"),
            "retrained_target_auc_max": target_auc,
            "retrained_control_auc_max": control_auc,
            "robust_erasure": _robust_erasure_from_auc(target_auc),
            "direction_stability_median": _nested(row, "direction_stability", "median"),
            "candidate_evaluator_agreement": _nested(
                row, "candidate_evaluator_agreement", "median"
            ),
            "crossfit_R_fixed_auc": _nested(row, "crossfit", "fixed", "aggregate", "R_auc_oriented"),
            "max_eigen_residual": max(
                [x["relative_residual"] for x in _nested(row, "hessian", "eigenpair_residuals") or []],
                default=None,
            ),
        }
        for k in all_ks:
            item[f"A_sub_{k}"] = _nested(row, "hessian", "A_subspace", str(k))
            item[f"A_sub_{k}_null_z"] = _nested(row, "hessian", "random_null", str(k), "z")
        per_evaluator = _nested(evaluation, "fixed", "per_evaluator") or []
        for floor in auc_floors:
            values = []
            for evaluator in per_evaluator:
                metric = orientation_robust_cs(
                    evaluator["target_pre"], evaluator["target_post"],
                    evaluator["control_pre"], evaluator["control_post"],
                    clean_auc_floor=floor,
                )
                if metric["R_auc_oriented"] is not None:
                    values.append(metric["R_auc_oriented"])
            item[_floor_column(floor)] = float(np.mean(values)) if values else None
        out.append(item)
    return pd.DataFrame(out), all_ks


def _cell_medians(df, target, predictor):
    use = df[
        df[target].notna() & df[predictor].notna() & (df["position"] == "last")
    ].copy()
    if use.empty:
        return use
    keys = ["model", "layer", "task", "arch"]
    return use.groupby(keys, as_index=False)[[target, predictor]].median()


def association(df, target, predictor, bootstrap):
    cells = _cell_medians(df, target, predictor)
    if len(cells) < 3:
        return {"n": len(cells), "rho": None, "reason": "too few cells"}
    rho, p = spearmanr(cells[predictor], cells[target])
    pairs = list(zip(cells[predictor], cells[target]))
    model_task_clusters = list(zip(cells["model"], cells["task"]))
    model_task_block = cluster_bootstrap_spearman(
        pairs, model_task_clusters, B=bootstrap, seed=0
    )
    model_block = cluster_bootstrap_spearman(
        pairs, list(cells["model"]), B=bootstrap, seed=1
    )
    return {
        "n": int(len(cells)),
        "n_model_task_blocks": int(cells[["model", "task"]].drop_duplicates().shape[0]),
        "rho": float(rho),
        "p_asymptotic": float(p),
        "model_task_block_bootstrap": model_task_block,
        "model_block_bootstrap": model_block,
    }


def leave_one_out(df, target, predictor, dimension):
    rows = []
    for held_out in sorted(df[dimension].dropna().unique()):
        sub = _cell_medians(df[df[dimension] != held_out], target, predictor)
        if len(sub) < 3:
            continue
        rho, p = spearmanr(sub[predictor], sub[target])
        rows.append({
            "held_out": held_out,
            "rho": float(rho),
            "p": float(p),
            "n": int(len(sub)),
        })
    return rows


def within_cell_association(df, target, predictor):
    use = df[
        df[target].notna() & df[predictor].notna() & (df["position"] == "last")
    ].copy()
    if use.empty:
        return {"rho": None, "n": 0}
    # Median across seeds first, then remove each model-layer-task mean. This
    # asks whether architecture differences within a cell move together.
    med = use.groupby(["model", "layer", "task", "arch"], as_index=False)[
        [target, predictor]
    ].median()
    outer = ["model", "layer", "task"]
    med["target_centered"] = med[target] - med.groupby(outer)[target].transform("mean")
    med["predictor_centered"] = med[predictor] - med.groupby(outer)[predictor].transform("mean")
    rho, p = spearmanr(med["predictor_centered"], med["target_centered"])
    return {"rho": float(rho), "p": float(p), "n": int(len(med))}


def layer_tables(layer_rows):
    inclusion = []
    baseline = []
    sweep = []
    for row in layer_rows:
        inclusion.append({
            "model": row["model"], "task": row["task"],
            "position": row.get("position", "last"), "layer": row["layer"],
            "status": row.get("status"), "reason": row.get("reason"),
            "detail": row.get("detail"),
            "selection_A_raw": _nested(
                row, "selection_diagnostic", "A_raw_mean_abs_cosine"
            ),
            "selection_candidate_accuracy": _nested(
                row, "selection_diagnostic", "candidate_accuracy"
            ),
        })
        if row.get("status") != "included":
            continue
        for method, record in row.get("baselines", {}).items():
            baseline.append({
                "model": row["model"], "task": row["task"], "layer": row["layer"],
                "position": row.get("position", "last"), "method": method,
                "R_fixed_auc": _nested(record, "fixed", "aggregate", "R_auc_oriented"),
                "C_fixed_auc": _nested(record, "fixed", "aggregate", "C_auc_oriented"),
                "S_fixed_auc": _nested(record, "fixed", "aggregate", "S_auc_oriented"),
                "retrained_target_auc_max": _nested(
                    record, "retrained_summary", "target", "max_oriented_roc_auc"
                ),
                "retrained_control_auc_max": _nested(
                    record, "retrained_summary", "control", "max_oriented_roc_auc"
                ),
                "distortion_test_fro": record.get("distortion_test_fro"),
            })
        for record in row.get("perturbation_sweep", []):
            sweep.append({
                "model": row["model"], "task": row["task"], "layer": row["layer"],
                "position": row.get("position", "last"),
                "method": record["method"], "magnitude": record["magnitude"],
                "matched_R_auc": _nested(record, "matched", "R_auc_oriented"),
                "split_R_auc": _nested(record, "split", "aggregate", "R_auc_oriented"),
                "matched_C_auc": _nested(record, "matched", "C_auc_oriented"),
                "split_C_auc": _nested(record, "split", "aggregate", "C_auc_oriented"),
                "distortion_test_fro": record.get("distortion_test_fro"),
            })
    return pd.DataFrame(inclusion), pd.DataFrame(baseline), pd.DataFrame(sweep)


def _plot_scatter(df, predictor, out_path):
    sub = df[
        df[predictor].notna() & df["robust_erasure"].notna()
        & (df["position"] == "last")
    ]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    for arch in ("linear", "mlp", "mka"):
        part = sub[sub["arch"] == arch]
        ax.scatter(part[predictor], part["robust_erasure"], s=34, alpha=0.75,
                   color=ARCH_COLOR[arch], marker=ARCH_MARKER[arch],
                   edgecolor="#333333", linewidth=0.35, label=arch.upper())
    if len(sub) >= 3:
        x = sub[predictor].to_numpy(float)
        y = sub["robust_erasure"].to_numpy(float)
        slope, intercept = np.polyfit(x, y, 1)
        line = np.linspace(x.min(), x.max(), 100)
        ax.plot(line, slope * line + intercept, "--", color=TREND, lw=1.5)
        rho, _ = spearmanr(x, y)
        ax.set_title(f"Post-edit redecodability vs Hessian subspace alignment (rho={rho:+.2f})")
    ax.set_xlabel(f"basis-invariant Hessian alignment ({predictor.replace('_', ' ')})")
    ax.set_ylabel("robust erasure score (fresh decoder; 1 = chance)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_baselines(candidate_df, baseline_df, out_path):
    rows = []
    deep = candidate_df[
        candidate_df["retrained_target_auc_max"].notna()
        & (candidate_df["position"] == "last")
    ]
    if not deep.empty:
        rows.append(("Candidate r=1", deep["retrained_target_auc_max"].to_numpy(float), CRIMSON))
    if not baseline_df.empty:
        for method, color in (
            ("random_rank1", "#C7C7C7"),
            ("random_distortion_matched", "#A6A6A6"),
            ("LEACE", SALMON),
            ("INLP", GRAY),
            ("RLACE_approx", "#6F6F6F"),
            ("supervised_linear_direction", "#B86B77"),
        ):
            vals = baseline_df[
                (baseline_df["method"] == method)
                & (baseline_df["position"] == "last")
            ]["retrained_target_auc_max"].dropna().to_numpy(float)
            if len(vals):
                label = method.replace("supervised_linear_direction", "Supervised direction")
                label = label.replace("RLACE_approx", "RLACE (repository approximation)")
                label = label.replace("random_rank1", "Random rank 1")
                label = label.replace("random_distortion_matched", "Random distortion matched")
                rows.append((label, vals, color))
    if not rows:
        return
    labels = [r[0] for r in rows]
    medians = [float(np.median(r[1])) for r in rows]
    lows = [float(np.percentile(r[1], 25)) for r in rows]
    highs = [float(np.percentile(r[1], 75)) for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(rows))
    ax.bar(x, medians, color=[r[2] for r in rows], edgecolor="#333333", linewidth=0.4)
    ax.errorbar(x, medians,
                yerr=[np.array(medians) - np.array(lows), np.array(highs) - np.array(medians)],
                fmt="none", ecolor="#333333", capsize=3, lw=1)
    ax.axhline(0.5, color=TREND, ls="--", lw=1.3, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("max post-edit oriented AUC across fresh decoders")
    ax.set_ylim(0.45, 1.02)
    ax.set_title("Fixed-evaluator damage is not treated as erasure without redecoding")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_sweep(sweep_df, out_path):
    if sweep_df.empty:
        return
    sub = sweep_df[sweep_df["position"] == "last"]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), sharey=True)
    for ax, method in zip(axes, ("AlterRep", "FGSM", "PGD")):
        part = sub[sub["method"] == method]
        if part.empty:
            continue
        agg = part.groupby("magnitude")[["matched_R_auc", "split_R_auc"]].median()
        ax.plot(agg.index, agg["matched_R_auc"], "o-", color=CRIMSON, label="matched")
        ax.plot(agg.index, agg["split_R_auc"], "s--", color=SALMON, label="split")
        ax.set_title(method)
        ax.set_xlabel("magnitude")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("median orientation-robust R")
    axes[0].legend(frameon=False)
    fig.suptitle("Evaluator coupling and saturation across perturbation strength")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_hessian_k(k_rows, out_path):
    if not k_rows:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    ks = [r["k"] for r in k_rows]
    rho_fixed = [r["rho_fixed"] for r in k_rows]
    rho_erasure = [r["rho_erasure"] for r in k_rows]
    ax.plot(ks, rho_fixed, "o-", color=GRAY, label="fixed-evaluator R")
    ax.plot(ks, rho_erasure, "s-", color=CRIMSON, label="fresh-decoder erasure")
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(ks)
    ax.set_xlabel("top-k Hessian subspace")
    ax.set_ylabel("Spearman rho")
    ax.set_title("Hessian conclusion across basis-invariant subspace sizes")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _json_native(obj):
    if isinstance(obj, dict):
        return {str(k): _json_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def main():
    args = parse_args()
    auc_floors = sorted(set(float(x.strip()) for x in args.auc_floors.split(",") if x.strip()))
    input_dir = Path(args.input_dir) if Path(args.input_dir).is_absolute() else PROJECT_ROOT / args.input_dir
    out_dir = (
        Path(args.out_dir) if args.out_dir and Path(args.out_dir).is_absolute()
        else PROJECT_ROOT / args.out_dir if args.out_dir
        else input_dir / "analysis"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows = _load_jsonl_files(input_dir.glob("*.candidates.jsonl"))
    layer_rows = _load_jsonl_files(input_dir.glob("*.layers.jsonl"))
    candidate_df, ks = candidate_table(candidate_rows, auc_floors) if candidate_rows else (pd.DataFrame(), [])
    inclusion_df, baseline_df, sweep_df = layer_tables(layer_rows)

    if not inclusion_df.empty:
        inclusion_df.to_csv(out_dir / "inclusion_matrix.csv", index=False)
    if not baseline_df.empty:
        baseline_df.to_csv(out_dir / "baseline_results.csv", index=False)
    if not sweep_df.empty:
        sweep_df.to_csv(out_dir / "perturbation_sweep.csv", index=False)
    if not candidate_df.empty:
        candidate_df.to_csv(out_dir / "candidate_results.csv", index=False)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "n_candidate_rows": len(candidate_rows),
        "n_layer_rows": len(layer_rows),
        "inclusion": {
            "status_counts": inclusion_df["status"].value_counts(dropna=False).to_dict()
            if not inclusion_df.empty else {},
            "exclusion_reasons": inclusion_df.loc[
                inclusion_df["status"] == "excluded", "reason"
            ].value_counts(dropna=False).to_dict() if not inclusion_df.empty else {},
        },
    }
    k_rows = []
    if not candidate_df.empty:
        preferred_k = 20 if 20 in ks else max(ks)
        preferred = f"A_sub_{preferred_k}"
        summary["associations"] = {
            "raw_A_vs_fixed_R": association(candidate_df, "R_fixed_auc", "A_raw", args.bootstrap),
            "subspace_A_vs_fixed_R": association(candidate_df, "R_fixed_auc", preferred, args.bootstrap),
            "subspace_A_vs_robust_erasure": association(
                candidate_df, "robust_erasure", preferred, args.bootstrap
            ),
            "within_cell_subspace_A_vs_fixed_R": within_cell_association(
                candidate_df, "R_fixed_auc", preferred
            ),
            "within_cell_subspace_A_vs_robust_erasure": within_cell_association(
                candidate_df, "robust_erasure", preferred
            ),
        }
        summary["leave_one_model_out"] = leave_one_out(
            candidate_df, "R_fixed_auc", preferred, "model"
        )
        summary["leave_one_task_out"] = leave_one_out(
            candidate_df, "R_fixed_auc", preferred, "task"
        )
        summary["direction_stability"] = {
            "n": int(candidate_df["direction_stability_median"].notna().sum()),
            "median": float(candidate_df["direction_stability_median"].median())
            if candidate_df["direction_stability_median"].notna().any() else None,
        }
        summary["candidate_evaluator_agreement"] = {
            "n": int(candidate_df["candidate_evaluator_agreement"].notna().sum()),
            "median": float(candidate_df["candidate_evaluator_agreement"].median())
            if candidate_df["candidate_evaluator_agreement"].notna().any() else None,
        }
        summary["auc_floor_sensitivity"] = {
            f"{floor:.2f}": association(
                candidate_df, _floor_column(floor), preferred, args.bootstrap
            )
            for floor in auc_floors
        }
        summary["eigenpair_residuals"] = {
            "n": int(candidate_df["max_eigen_residual"].notna().sum()),
            "median_max_residual": float(candidate_df["max_eigen_residual"].median())
            if candidate_df["max_eigen_residual"].notna().any() else None,
            "p95_max_residual": float(candidate_df["max_eigen_residual"].quantile(0.95))
            if candidate_df["max_eigen_residual"].notna().any() else None,
        }
        excluded_selection = inclusion_df[
            (inclusion_df["status"] == "excluded")
            & (inclusion_df["position"] == "last")
            & inclusion_df["selection_A_raw"].notna()
        ] if not inclusion_df.empty else pd.DataFrame()
        included_selection = candidate_df[
            (candidate_df["position"] == "last")
            & (candidate_df["arch"] == "linear")
            & (candidate_df["seed"] == 1000)
            & candidate_df["A_raw"].notna()
        ]
        summary["selection_diagnostic"] = {
            "included_n": int(len(included_selection)),
            "included_A_raw_median": float(included_selection["A_raw"].median())
            if len(included_selection) else None,
            "included_accuracy_median": float(included_selection["candidate_accuracy"].median())
            if len(included_selection) else None,
            "excluded_n": int(len(excluded_selection)),
            "excluded_A_raw_median": float(excluded_selection["selection_A_raw"].median())
            if len(excluded_selection) else None,
            "excluded_accuracy_median": float(
                excluded_selection["selection_candidate_accuracy"].median()
            ) if len(excluded_selection) else None,
        }
        for k in ks:
            fixed = association(candidate_df, "R_fixed_auc", f"A_sub_{k}", args.bootstrap)
            erasure = association(candidate_df, "robust_erasure", f"A_sub_{k}", args.bootstrap)
            k_rows.append({
                "k": k,
                "rho_fixed": fixed.get("rho"),
                "rho_erasure": erasure.get("rho"),
                "n_fixed": fixed.get("n"),
                "n_erasure": erasure.get("n"),
            })
        pd.DataFrame(k_rows).to_csv(out_dir / "hessian_k_sensitivity.csv", index=False)

        _plot_scatter(candidate_df, preferred, out_dir / "fig_v3_redecodability.png")
        _plot_baselines(candidate_df, baseline_df, out_dir / "fig_v3_baselines.png")
        _plot_hessian_k(k_rows, out_dir / "fig_v3_hessian_sensitivity.png")
    _plot_sweep(sweep_df, out_dir / "fig_v3_perturbation_sweep.png")

    if not baseline_df.empty:
        summary["baseline_medians"] = baseline_df.groupby("method")[[
            "R_fixed_auc", "retrained_target_auc_max", "retrained_control_auc_max"
        ]].median(numeric_only=True).to_dict(orient="index")
    if not sweep_df.empty:
        summary["perturbation_saturation"] = {
            f"{method}@{magnitude}": {
                "n": int(len(group)),
                "matched_R_median": float(group["matched_R_auc"].median()),
                "split_R_median": float(group["split_R_auc"].median()),
                "matched_saturation_fraction": float((group["matched_R_auc"] >= 0.98).mean()),
                "split_saturation_fraction": float((group["split_R_auc"] >= 0.98).mean()),
            }
            for (method, magnitude), group in sweep_df.groupby(["method", "magnitude"])
        }

    summary_path = out_dir / "robustness_summary.json"
    summary_path.write_text(json.dumps(_json_native(summary), indent=2, sort_keys=True))
    print(f"Loaded {len(candidate_rows)} candidate rows and {len(layer_rows)} layer rows")
    print(f"Analysis: {out_dir}")
    print(f"Summary: {summary_path}")
    if not candidate_rows:
        print("No candidate rows were available; only the inclusion matrix was produced.")


if __name__ == "__main__":
    main()
