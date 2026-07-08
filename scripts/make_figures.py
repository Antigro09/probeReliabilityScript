"""
Publication figures for the v2 result.

Reads results/benchmark_v2/*.jsonl (schema v2, repaired) and, if present, the
synthetic head-to-head (results/ws5/headtohead_*.json), and writes PNGs to
results/figures/.

Palette matches the paper's house style:
    Linear  -> gray      MLP -> salmon      MKA -> crimson    trend -> red dashed

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

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Paper house palette.
GRAY = "#808080"
SALMON = "#E8A0A0"
CRIMSON = "#D1495B"
TREND = "#D62728"   # red dashed trend/reference lines

ARCH_COLOR = {"linear": GRAY, "mlp": SALMON, "mka": CRIMSON}
ARCH_LABEL = {"linear": "Linear", "mlp": "MLP", "mka": "MKA"}
ARCH_MARKER = {"linear": "o", "mlp": "s", "mka": "^"}


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


def _trend(ax, xs, ys):
    """Red dashed least-squares trend line over the facet's points."""
    if len(xs) < 3:
        return
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    b, a = np.polyfit(xs, ys, 1)
    xline = np.linspace(xs.min(), xs.max(), 50)
    ax.plot(xline, b * xline + a, "--", color=TREND, lw=1.6, zorder=1)


def fig1_scatter(cm):
    tasks = sorted({k[2] for k in cm})
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 4), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        sub = {k: v for k, v in cm.items() if k[2] == task and "R_cand" in v}
        allx, ally = [], []
        for arch in ARCH_COLOR:
            xs = [v["A"] for k, v in sub.items() if k[3] == arch]
            ys = [v["R_cand"] for k, v in sub.items() if k[3] == arch]
            ax.scatter(xs, ys, s=34, alpha=0.85, color=ARCH_COLOR[arch],
                       marker=ARCH_MARKER[arch], label=ARCH_LABEL[arch],
                       edgecolor="#333333", linewidth=0.4, zorder=3)
            allx += xs; ally += ys
        _trend(ax, allx, ally)
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
        cols.append(CRIMSON if p < 0.05 else GRAY)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(range(len(vals)), vals, color=cols, width=0.6, edgecolor="#333333", linewidth=0.4)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(0.5, color=TREND, lw=1.3, ls="--", label="prereg threshold (0.5)")
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
    # alignment = crimson (matches the paper's "MKA"/alignment line); reliability = salmon
    ax.plot([ARCH_LABEL[a] for a in archs], medA, "o-", color=CRIMSON, lw=2, label="median A")
    ax2.plot([ARCH_LABEL[a] for a in archs], medR, "s--", color=SALMON, lw=2, label="median R_cand")
    ax.set_ylabel("median alignment A", color=CRIMSON); ax.tick_params(axis="y", labelcolor=CRIMSON)
    ax2.set_ylabel("median R_cand", color=SALMON); ax2.tick_params(axis="y", labelcolor="#B06A6A")
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
        cols = [GRAY, GRAY, GRAY, CRIMSON]   # highlight the repaired metric
        ax.bar(order, vals, color=cols, width=0.6, edgecolor="#333333", linewidth=0.4)
        ax.axhline(0.5, color=TREND, ls="--", lw=1.3)
        fam = Path(f).stem.replace("headtohead_", "")
        dt = d.get("delta_tau_R_cand_minus_v1max")
        ax.set_title(f"synthetic ({fam})\nKendall tau vs ground truth  (delta={dt:.2f})" if dt is not None else fam, fontsize=9)
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
