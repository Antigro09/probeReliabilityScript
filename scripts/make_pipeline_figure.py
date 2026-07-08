"""
Methods pipeline schematic (fig0).

A minimal, reproducible flow diagram of the v2 methodology. Flow reads
left-to-right across the top (data -> extraction -> candidate probe), then the
candidate probe forks DOWN into the two objects under study — the cheap
predictor A and the repaired reliability target R_cand — which converge into the
pre-registered test. A separate lane defines the synthetic validation.

Palette matches the paper's data figures (gray / salmon / crimson). Outputs both
results/figures/fig0_pipeline.png and .svg (vector, for the manuscript).

Usage: python -m scripts.make_pipeline_figure
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

GRAY = "#808080"
SALMON = "#E8A0A0"
CRIMSON = "#D1495B"
INK = "#222222"
PRE_FILL = "#F2F2F2"
CRIMSON_FILL = "#F7DBE0"
SALMON_FILL = "#FBECEC"


def box(ax, cx, cy, w, h, text, *, edge=GRAY, fill="white", lw=1.6,
        fontsize=9, bold=False, dashed=False, textcolor=INK):
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.10",
                       linewidth=lw, edgecolor=edge, facecolor=fill,
                       linestyle=("--" if dashed else "-"), zorder=2)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            color=textcolor, zorder=3, weight=("bold" if bold else "normal"))
    return {"cx": cx, "cy": cy, "w": w, "h": h}


def arrow(ax, a, b, *, color=INK, lw=1.8, label=None, side="mid",
          a_side="auto", b_side="auto"):
    """Arrow between box edges a -> b (dicts from box())."""
    def anchor(bx, other, which):
        if which == "auto":
            dx, dy = other["cx"] - bx["cx"], other["cy"] - bx["cy"]
            which = ("right" if abs(dx) >= abs(dy) and dx > 0 else
                     "left" if abs(dx) >= abs(dy) else
                     "top" if dy > 0 else "bottom")
        return {"right": (bx["cx"] + bx["w"] / 2, bx["cy"]),
                "left": (bx["cx"] - bx["w"] / 2, bx["cy"]),
                "top": (bx["cx"], bx["cy"] + bx["h"] / 2),
                "bottom": (bx["cx"], bx["cy"] - bx["h"] / 2)}[which]
    x1, y1 = anchor(a, b, a_side)
    x2, y2 = anchor(b, a, b_side)
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=15, lw=lw, color=color,
                                 shrinkA=1, shrinkB=1, zorder=1))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, label, ha="center",
                va="bottom", fontsize=7.5, style="italic", color=color, zorder=4)


def main():
    fig, ax = plt.subplots(figsize=(15.5, 9.4))
    ax.set_xlim(0, 16); ax.set_ylim(0, 10.4); ax.axis("off")

    ax.text(8, 10.05, "Methods pipeline: does the cheap alignment score $A$ "
            "predict repaired probe reliability $R_{cand}$?",
            ha="center", va="center", fontsize=13.5, weight="bold")

    # ---- legend (defines every element) ----
    lx = 0.5
    for swab, lab in [(GRAY, "shared preprocessing"), (CRIMSON, "predictor path ($A$)"),
                      (SALMON, "reliability-target path ($R_{cand}$)")]:
        ax.add_patch(FancyBboxPatch((lx, 9.35), 0.34, 0.22, boxstyle="round,pad=0.01,rounding_size=0.05",
                                    edgecolor=swab, facecolor="white", lw=2.2))
        ax.text(lx + 0.44, 9.46, lab, va="center", fontsize=8.2)
        lx += 0.30 + 0.115 * len(lab)
    ax.plot([lx + 0.05, lx + 0.39], [9.46, 9.46], "--", color=GRAY, lw=1.6)
    ax.text(lx + 0.48, 9.46, "validation ($A$ not used)", va="center", fontsize=8.2)

    # ---- top row: data -> extraction -> candidate probe ----
    b1 = box(ax, 1.6, 7.9, 2.5, 1.05, "Text corpus\n(SVA · SST-2)", fill=PRE_FILL)
    b2 = box(ax, 4.7, 7.9, 2.9, 1.05, "Deduplicate +\nclass-balanced\n4-way split", fill=PRE_FILL)
    b3 = box(ax, 8.0, 7.9, 3.0, 1.05, "Frozen LM →\nlast-token hidden state\n(per layer, z-scored)", fill=PRE_FILL)
    p0 = box(ax, 11.9, 7.9, 3.4, 1.05, "Train candidate probe\n(candidate fold)\n{Linear, MLP, MKA} × 20 seeds",
             edge=INK, lw=2.0)
    arrow(ax, b1, b2); arrow(ax, b2, b3); arrow(ax, b3, p0)

    # ---- 4-way split inset (fills the space; defines the split visually) ----
    seg = [("candidate", 0.40, "#6E6E6E"), ("evaluator", 0.20, "#8E8E8E"),
           ("intervention", 0.30, "#AEAEAE"), ("test", 0.10, "#CCCCCC")]
    bx, by, bw, bh = 1.3, 5.35, 6.0, 0.55
    ax.text(bx, by + bh + 0.42, "4-way disjoint split  (per model × task)",
            fontsize=9.5, weight="bold")
    xc = bx
    for name, frac, col in seg:
        w = frac * bw
        ax.add_patch(Rectangle((xc, by), w, bh, facecolor=col, edgecolor="white", lw=1.4))
        ax.text(xc + w / 2, by + bh / 2, f"{int(frac*100)}%", ha="center", va="center",
                fontsize=8.5, color="white", weight="bold")
        ax.text(xc + w / 2, by - 0.26, name, ha="center", va="top", fontsize=8)
        xc += w

    # ---- fork DOWN into predictor (A) and target (R_cand) ----
    bA = box(ax, 9.9, 5.55, 3.3, 1.55,
             "$A$   (cheap predictor)\n$\\mathrm{mean}\\,|\\cos(w,\\,v_i)|$\nover the top-20 eigenvectors\n$v_i$ of the probe-loss Hessian\n— no intervention —",
             edge=CRIMSON, fill=CRIMSON_FILL, lw=2.0, fontsize=8.6)
    bR = box(ax, 13.7, 5.55, 3.7, 1.55,
             "$R_{cand}$   (reliability target)\nerase candidate direction\n$d_c$:  $X(I-QQ^{\\top})$ on interv. fold,\nscore with 5 independent bagged\nevaluators (eval fold): mean HM($C$,$S$)",
             edge=SALMON, fill=SALMON_FILL, lw=2.0, fontsize=8.4)
    arrow(ax, p0, bA, color=CRIMSON, a_side="bottom", b_side="top", label="probe params $w$")
    arrow(ax, p0, bR, color=SALMON, a_side="bottom", b_side="top", label="direction $d_c$")

    # ---- converge into the preregistered test ----
    rho = box(ax, 11.8, 3.35, 4.2, 0.95,
              "Preregistered test:  Spearman $\\rho(A,\\ R_{cand})$   (RP1)",
              edge=INK, lw=2.0, bold=True, fontsize=9.5)
    arrow(ax, bA, rho, color=CRIMSON, a_side="bottom", b_side="top")
    arrow(ax, bR, rho, color=SALMON, a_side="bottom", b_side="top")

    # ---- validation lane (full-width band, clearly separated) ----
    box(ax, 8.0, 1.55, 15.0, 1.5,
        "Synthetic validation   ($A$ is never computed)   — validates the $R_{cand}$ target\n"
        "plant a causal axis $v_c$, a control feature $v_e$, and a spurious shortcut $v_s$  →  ground truth is known by construction.\n"
        "$R_{cand}$ ranks probes by recovery of $v_c$ (Kendall $\\tau\\!\\approx\\!0.74$), which the v1 max-over-methods metric cannot  ⇒  $R_{cand}$ is a valid, non-circular target.",
        edge=GRAY, fill="#FAFAFA", lw=1.6, dashed=True, fontsize=8.6)

    # ---- footnote: define the folds ----
    ax.text(8, 0.55, "4-way split (disjoint):   candidate = train the probe   ·   "
            "evaluator = train the certifiers   ·   intervention = apply & measure the edit   ·   "
            "test = probe accuracy", ha="center", va="center", fontsize=8, color="#555555")
    ax.text(8, 0.22, "$C$ = completeness (concept removed) · $S$ = selectivity (control preserved) · "
            "HM = harmonic mean · $Q$ = orthonormal basis of $d_c$",
            ha="center", va="center", fontsize=8, color="#555555")

    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"fig0_pipeline.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'fig0_pipeline.png'} and .svg")


if __name__ == "__main__":
    main()
