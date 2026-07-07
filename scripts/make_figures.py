"""
Publication figures for the v2 result.

Reads results/benchmark_v2/*.jsonl (schema v2, repaired) and, if present, the
synthetic head-to-head (results/ws5/headtohead_*.json), and writes PNGs to
results/figures/.

Figures:
  fig1_A_vs_Rcand   - the null result: A vs repaired R_cand (cell medians), per task.
  fig2_ladder       - rho(A, rung) across the decomposition ladder (the money plot).
  fig3_arch         - per-architecture A and R_cand (they move oppositely).
  fig4_synth        - synthetic head-to-head: Kendall tau of each rung vs ground truth.

Usage: python -m scripts.make_figures
"""
from __future__ import annotations

import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

ARCH_COLOR = {"linear": "#4C72B0", "mlp": "#DD8452", "mka": "#55A868"}


def load_rows():
    rows = []
    for f in glob.glob(str(PROJECT_ROOT / "results" / "benchmark_v2" / "*.jsonl")):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def cell_medians(rows):
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = (r["model"], r["layer"], r["task"], r["arch"])
        for key in ("A", "R_cand", "R_excl", "R_v1_max", "R_v1_mean", "accuracy"):
            v = r.get(key)
            if v is not None:
                cells[k][key].append(v)
    return {k: {key: statistics.median(vs) for key, vs in d.items()} for k, d in cells.items()}


def _rho(cm, x, y):
    pts = [(c[x], c[y]) for c in cm.values() if x in c and y in c]
    if len(pts) < 3:
        return float("nan"), float("nan"), len(pts)
    r, p = spearmanr([a for a, _ in pts], [b for _, b in pts])
    return float(r), float(p), len(pts)


def fig1_scatter(cm):
    tasks = sorted({k[2] for k in cm})
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 4), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        sub = {k: v for k, v in cm.items() if k[2] == task and "R_cand" in v}
        for arch in ARCH_COLOR:
            xs = [v["A"] for k, v in sub.items() if k[3] == arch]
            ys = [v["R_cand"] for k, v in sub.items() if k[3] == arch]
            ax.scatter(xs, ys, s=28, alpha=0.75, color=ARCH_COLOR[arch], label=arch, edgecolor="none")
        r, p, n = _rho(sub, "A", "R_cand")
        ax.set_title(f"{task}   rho={r:.2f} (p={p:.2f}, n={n})", fontsize=10)
        ax.set_xlabel("alignment A"); ax.set_ylabel("repaired reliability R_cand")
        ax.grid(alpha=0.25)
    axes[0][0].legend(frameon=False, fontsize=8, title="probe")
    fig.suptitle("Alignment A does NOT predict repaired reliability (per-cell medians)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_A_vs_Rcand.png", dpi=150); plt.close(fig)


def fig2_ladder(cm):
    rungs = [("R_v1_max", "v1 max-over-5\n(broken)"), ("R_excl", "R_excl\n(layer-level)"),
             ("accuracy", "test accuracy"), ("R_cand", "R_cand\n(repaired, validated)")]
    vals, labels, cols = [], [], []
    for key, lab in rungs:
        r, p, n = _rho(cm, "A", key)
        vals.append(r); labels.append(f"{lab}\nrho={r:+.2f}" + ("*" if p < 0.05 else ""))
        cols.append("#C44E52" if p < 0.05 else "#8C8C8C")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(range(len(vals)), vals, color=cols, width=0.6)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(0.5, color="green", lw=0.8, ls="--", label="prereg threshold (0.5)")
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Spearman rho(A, target)"); ax.set_ylim(-0.5, 0.6)
    ax.set_title("What A correlates with: significant & NEGATIVE vs v1, null vs repaired")
    ax.legend(frameon=False, fontsize=8); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "fig2_ladder.png", dpi=150); plt.close(fig)


def fig3_arch(rows):
    fig, ax = plt.subplots(figsize=(6, 4))
    archs = ["linear", "mlp", "mka"]
    medA = [statistics.median([r["A"] for r in rows if r["arch"] == a]) for a in archs]
    medR = [statistics.median([r["R_cand"] for r in rows if r["arch"] == a and r["R_cand"] is not None]) for a in archs]
    ax2 = ax.twinx()
    ax.plot(archs, medA, "o-", color="#4C72B0", label="median A")
    ax2.plot(archs, medR, "s--", color="#C44E52", label="median R_cand")
    ax.set_ylabel("median A", color="#4C72B0"); ax2.set_ylabel("median R_cand", color="#C44E52")
    ax.set_title("Higher capacity: more reliable (R_cand up) but less aligned (A down)")
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "fig3_arch.png", dpi=150); plt.close(fig)


def fig4_synth():
    files = glob.glob(str(PROJECT_ROOT / "results" / "ws5" / "headtohead_*.json"))
    if not files:
        return False
    fig, axes = plt.subplots(1, len(files), figsize=(4.5 * len(files), 4), squeeze=False)
    for ax, f in zip(axes[0], sorted(files)):
        d = json.load(open(f))
        tau = d.get("tau", {})
        order = ["R_v1_max", "R_v1_mean", "R_excl", "R_cand"]
        vals = [tau.get(k, float("nan")) for k in order]
        cols = ["#8C8C8C", "#8C8C8C", "#8C8C8C", "#55A868"]
        ax.bar(order, vals, color=cols, width=0.6)
        ax.axhline(0.5, color="green", ls="--", lw=0.8)
        fam = Path(f).stem.replace("headtohead_", "")
        dt = d.get("delta_tau_R_cand_minus_v1max")
        ax.set_title(f"synthetic ({fam})\nKendall tau vs ground truth  (delta={dt:.2f})" if dt else fam, fontsize=9)
        ax.set_ylabel("tau(rung, true recovery)"); ax.tick_params(axis="x", labelrotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Synthetic validation: R_cand tracks ground-truth recovery; v1 max does not", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig4_synth.png", dpi=150); plt.close(fig)
    return True


def main():
    rows = load_rows()
    cm = cell_medians(rows)
    models = sorted({k[0] for k in cm})
    print(f"loaded {len(rows)} rows, {len(cm)} cells, models: {models}")
    fig1_scatter(cm)
    fig2_ladder(cm)
    fig3_arch(rows)
    got_synth = fig4_synth()
    print(f"wrote figures to {OUT}:")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}")
    if not got_synth:
        print("  (fig4_synth skipped: run scripts.ws5_run first)")


if __name__ == "__main__":
    main()
