"""Publication figures generated strictly from materialized revision rows.

This module intentionally imports no model, probe, cache, or intervention code.
The three public renderers validate complete saved-row artifacts and export a
vector PDF plus a 300-DPI PNG with the same deterministic plotting logic.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .analysis import (
    HARD_FAILURE_STATUSES,
    SCORE_NULL_STATUSES,
    AnalysisValidationError,
    load_rows,
)

OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "pink": "#CC79A7",
    "black": "#000000",
    "gray": "#777777",
    "light_gray": "#C7C7C7",
}

LOCKED_EPSILONS = (
    0.0,
    0.001953125,
    0.00390625,
    0.0078125,
    0.015625,
    0.03125,
    0.0625,
    0.125,
    0.25,
    0.5,
)
LOCKED_METHODS = ("fgsm", "pgd")
LOCKED_CONDITIONS = ("matched", "split")
LOCKED_PAIR_SEEDS = frozenset(range(5))
LOCKED_MODELS = ("pythia", "gpt2", "bert", "qwen", "gemma", "llama")
LOCKED_TASKS = ("sva", "sst2")
CANDIDATE_EDIT_OBJECTS = frozenset(("candidate", "dcand_crossfit"))
CANDIDATE_EDIT_ID = re.compile(
    r"^dcand_crossfit(?P<separator>[:-])(?P<architecture>linear|mlp|mka)"
    r"(?P=separator)seed-?(?P<seed>\d+)$"
)


class FigureValidationError(ValueError):
    """Raised when saved rows cannot support a locked revision figure."""


@dataclass(frozen=True)
class FigureArtifacts:
    """The two publication exports and audit metadata for one figure."""

    stem: str
    pdf_path: Path
    png_path: Path
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stem": self.stem,
            "pdf_path": str(self.pdf_path),
            "png_path": str(self.png_path),
            "metadata": dict(self.metadata),
        }


def _publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.45,
            "grid.linestyle": "-",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "lines.linewidth": 1.3,
        }
    )


def _column(frame: pd.DataFrame, *candidates: str) -> str:
    for candidate in candidates:
        if candidate in frame:
            return candidate
    raise FigureValidationError(
        f"saved rows require one of these columns: {list(candidates)!r}"
    )


def _saved_rows(source: str | Path) -> pd.DataFrame:
    if not isinstance(source, (str, Path)):
        raise FigureValidationError("figure input must be a saved row-file path")
    path = Path(source)
    try:
        frame = load_rows(path)
    except AnalysisValidationError as exc:
        raise FigureValidationError(str(exc)) from exc
    if frame.empty:
        raise FigureValidationError(f"saved row artifact is empty: {path}")
    if "status" not in frame:
        raise FigureValidationError("saved rows require an explicit status column")
    statuses = frame["status"].astype(str)
    hard = frame.loc[statuses.isin(HARD_FAILURE_STATUSES)]
    unknown = frame.loc[~statuses.isin({"ok", *SCORE_NULL_STATUSES})]
    if not hard.empty or not unknown.empty:
        raise FigureValidationError(
            "saved rows include hard or unknown failure statuses; figures cannot "
            "exclude execution failures"
        )
    score_nulls = frame.loc[statuses.isin(SCORE_NULL_STATUSES)]
    if not score_nulls.empty:
        for column in ("failure_stage", "failure_reason"):
            if column not in score_nulls or score_nulls[column].fillna("").astype(str).str.strip().eq("").any():
                raise FigureValidationError(
                    f"score-null rows require an explicit {column}"
                )
    return frame


def _finite_unit_interval(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise FigureValidationError(f"{column} contains non-finite values")
        if ((values < 0.0) | (values > 1.0)).any():
            raise FigureValidationError(f"{column} must lie in [0, 1]")


def _save(
    figure: plt.Figure,
    output_directory: str | Path,
    stem: str,
    metadata: Mapping[str, Any],
) -> FigureArtifacts:
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    pdf_path = directory / f"{stem}.pdf"
    png_path = directory / f"{stem}.png"
    figure.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        metadata={"Title": stem, "Creator": "reviewer_revision.figures"},
    )
    figure.savefig(
        png_path,
        format="png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    plt.close(figure)
    return FigureArtifacts(stem, pdf_path, png_path, dict(metadata))


def _matched_pair_rows(frame: pd.DataFrame) -> pd.DataFrame:
    model = _column(frame, "model", "model_key")
    task = _column(frame, "task", "task_key")
    method = _column(frame, "method")
    condition = _column(frame, "condition", "scoring_condition")
    damage = _column(
        frame,
        "target_damage_C",
        "mean_target_damage_C",
        "C",
        "target_damage",
    )
    required = ("layer", "pair_seed")
    missing = [name for name in required if name not in frame]
    if missing:
        raise FigureValidationError(f"matched/split rows lack columns: {missing!r}")
    primary_requested = frame.loc[
        frame[method].astype(str).str.lower() == "alterrep"
    ].copy()
    if primary_requested.empty:
        raise FigureValidationError("matched/split rows contain no AlterRep units")
    result = pd.DataFrame(
        {
            "model": primary_requested[model].astype(str),
            "task": primary_requested[task].astype(str),
            "layer": primary_requested["layer"].astype(int),
            "pair_seed": primary_requested["pair_seed"].astype(int),
            "condition": primary_requested[condition].astype(str).str.lower(),
            "damage": pd.to_numeric(primary_requested[damage], errors="coerce"),
            "status": primary_requested["status"].astype(str),
        }
    )
    if "depth_position" in primary_requested:
        result["depth_position"] = pd.to_numeric(
            primary_requested["depth_position"], errors="coerce"
        )
    keys = ["model", "task", "layer", "pair_seed"]
    if result.duplicated([*keys, "condition"]).any():
        raise FigureValidationError("matched/split rows contain duplicate pair conditions")
    condition_sets = result.groupby(keys)["condition"].agg(frozenset)
    if (condition_sets != frozenset(LOCKED_CONDITIONS)).any():
        raise FigureValidationError("every AlterRep pair requires matched and split rows")
    seed_sets = result.groupby(["model", "task", "layer"])["pair_seed"].agg(
        lambda values: frozenset(int(value) for value in values)
    )
    if (seed_sets != LOCKED_PAIR_SEEDS).any():
        raise FigureValidationError("every matched/split cell requires pair seeds 0 through 4")
    requested_cells = result[["model", "task", "layer"]].drop_duplicates()
    complete_mask = result.groupby(keys, sort=False, dropna=False)["status"].transform(
        lambda values: bool((values.astype(str) == "ok").all())
    )
    result = result.loc[complete_mask].copy()
    if result.empty:
        raise FigureValidationError("no complete AlterRep pairs remain for plotting")
    _finite_unit_interval(result, ("damage",))
    valid_cells = result[["model", "task", "layer"]].drop_duplicates()
    valid_cell_keys = set(valid_cells.itertuples(index=False, name=None))
    excluded_cells = [
        {"model": key[0], "task": key[1], "layer": int(key[2])}
        for key in requested_cells.itertuples(index=False, name=None)
        if tuple(key) not in valid_cell_keys
    ]
    paired = (
        result.set_index([*keys, "condition"])["damage"]
        .unstack("condition")
        .reset_index()
    )
    paired["gap"] = paired["matched"] - paired["split"]
    if "depth_position" in result:
        depth_counts = result.groupby(["model", "task", "layer"])[
            "depth_position"
        ].nunique()
        if (depth_counts != 1).any():
            raise FigureValidationError("depth position changes within a cell")
        depths = result.groupby(
            ["model", "task", "layer"], as_index=False
        )["depth_position"].first()
        paired = paired.merge(
            depths,
            on=["model", "task", "layer"],
            validate="many_to_one",
        )
    else:
        paired["depth_position"] = 0
        for _, block in paired.groupby(["model", "task"], sort=True):
            layer_order = {
                layer: index + 1
                for index, layer in enumerate(sorted(block["layer"].unique()))
            }
            paired.loc[block.index, "depth_position"] = block["layer"].map(layer_order)
    paired["depth_position"] = paired["depth_position"].astype(int)
    paired.attrs.update(
        {
            "requested_cells": len(requested_cells),
            "analyzed_cells": len(valid_cells),
            "excluded_cells": excluded_cells,
        }
    )
    return paired


def _depth_bootstrap(
    paired: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if draws <= 0:
        raise FigureValidationError("bootstrap_draws must be positive")
    rng = np.random.default_rng(seed)
    depths = np.array(sorted(paired["depth_position"].unique()), dtype=int)
    means: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for depth in depths:
        subset = paired.loc[paired["depth_position"] == depth]
        groups = [
            group["gap"].to_numpy(dtype=float)
            for _, group in subset.groupby(["model", "task"], sort=True)
        ]
        if not groups or len(groups) > 12:
            raise FigureValidationError(
                f"depth {depth} has an invalid available-block count: {len(groups)}"
            )
        block_means = np.array([values.mean() for values in groups])
        means.append(float(block_means.mean()))
        estimates = np.empty(draws, dtype=float)
        for draw in range(draws):
            sampled_blocks = rng.integers(0, len(groups), size=len(groups))
            sampled_means = [
                float(
                    rng.choice(groups[index], size=len(groups[index]), replace=True).mean()
                )
                for index in sampled_blocks
            ]
            estimates[draw] = float(np.mean(sampled_means))
        low, high = np.quantile(estimates, (0.025, 0.975))
        lows.append(float(low))
        highs.append(float(high))
    return depths, np.array(means), np.array(lows), np.array(highs)


def create_expanded_matched_split_figure(
    rows_path: str | Path,
    output_directory: str | Path,
    *,
    bootstrap_draws: int = 10_000,
    bootstrap_seed: int = 20260830,
    stem: str = "fig_circularity_expanded",
) -> FigureArtifacts:
    """Render paired cell damage and block-aware depth trajectories."""

    _publication_style()
    paired = _matched_pair_rows(_saved_rows(rows_path))
    paired_scope = dict(paired.attrs)
    cells = (
        paired.groupby(
            ["model", "task", "layer", "depth_position"],
            sort=True,
            as_index=False,
        )[["matched", "split", "gap"]]
        .mean()
    )
    block_sizes = cells.groupby(["model", "task"])["layer"].nunique()
    requested_cell_count = int(paired_scope["requested_cells"])
    requested_depths_per_block = requested_cell_count // 12
    if (
        len(block_sizes) != 12
        or requested_cell_count not in (36, 60)
        or (block_sizes > requested_depths_per_block).any()
        or (block_sizes < 1).any()
    ):
        raise FigureValidationError(
            "expanded matched/split figure lost a complete model-task block or "
            "does not match the locked requested grid"
        )
    if set(cells["model"]) != set(LOCKED_MODELS) or set(cells["task"]) != set(
        LOCKED_TASKS
    ):
        raise FigureValidationError(
            "matched/split figure requires the six locked models and tasks"
        )
    model_order = {model: index for index, model in enumerate(LOCKED_MODELS)}
    task_order = {task: index for index, task in enumerate(LOCKED_TASKS)}
    cells["_model_order"] = cells["model"].map(model_order)
    cells["_task_order"] = cells["task"].map(task_order)
    cells = cells.sort_values(
        ["_model_order", "_task_order", "depth_position"], kind="mergesort"
    ).drop(columns=["_model_order", "_task_order"]).reset_index(drop=True)
    blocks = list(cells.groupby(["model", "task"], sort=False))

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 3.05),
        gridspec_kw={"width_ratios": (1.65, 1.0), "wspace": 0.27},
    )
    paired_axis, depth_axis = axes

    x = np.arange(len(cells), dtype=float)
    for position, row in cells.iterrows():
        paired_axis.plot(
            [position - 0.13, position + 0.13],
            [row["matched"], row["split"]],
            color=OKABE_ITO["gray"],
            linewidth=0.65,
            zorder=1,
        )
    paired_axis.scatter(
        x - 0.13,
        cells["matched"],
        s=11,
        marker="o",
        color=OKABE_ITO["vermillion"],
        edgecolors="white",
        linewidths=0.25,
        label="Matched",
        zorder=3,
    )
    paired_axis.scatter(
        x + 0.13,
        cells["split"],
        s=13,
        marker="s",
        facecolors="white",
        edgecolors=OKABE_ITO["blue"],
        linewidths=0.8,
        label="Split",
        zorder=3,
    )
    centers: list[float] = []
    labels: list[str] = []
    offset = 0
    for (model, task), block in blocks:
        width = len(block)
        centers.append(offset + (width - 1) / 2)
        labels.append(f"{model}\n{task.upper()}")
        offset += width
        if offset < len(cells):
            paired_axis.axvline(offset - 0.5, color="#B5B5B5", linewidth=0.45)
    paired_axis.set_xticks(centers)
    paired_axis.set_xticklabels(labels, rotation=45, ha="right")
    paired_axis.set_ylabel("AlterRep target damage, $C$")
    paired_axis.set_xlabel("Model-task block; sampled layers ordered by depth")
    paired_axis.set_ylim(-0.03, 1.03)
    paired_axis.legend(loc="lower left", ncol=2)
    if paired_scope["excluded_cells"]:
        paired_axis.text(
            0.99,
            0.02,
            (
                f"{len(paired_scope['excluded_cells'])} requested cell "
                "non-estimable at the locked floor"
            ),
            transform=paired_axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.2,
            color=OKABE_ITO["gray"],
        )
    paired_axis.text(
        -0.09,
        1.03,
        "A",
        transform=paired_axis.transAxes,
        fontweight="bold",
        va="bottom",
    )

    for _, block in cells.groupby(["model", "task"], sort=True):
        depth_axis.plot(
            block["depth_position"],
            block["gap"],
            color=OKABE_ITO["light_gray"],
            linewidth=0.8,
            marker="o",
            markersize=2.2,
            zorder=1,
        )
    depths, means, lows, highs = _depth_bootstrap(
        paired,
        draws=bootstrap_draws,
        seed=bootstrap_seed,
    )
    depth_axis.errorbar(
        depths,
        means,
        yerr=np.vstack((means - lows, highs - means)),
        color=OKABE_ITO["black"],
        marker="D",
        markersize=4.0,
        capsize=2.5,
        linewidth=1.5,
        label="Equal-block mean (95% interval)",
        zorder=3,
    )
    depth_axis.axhline(0.0, color=OKABE_ITO["gray"], linewidth=0.65, linestyle="--")
    depth_axis.set_xticks(depths)
    depth_axis.set_xlabel("Sampled depth position")
    depth_axis.set_ylabel("Matched - split damage")
    depth_axis.set_ylim(
        min(-0.03, float(lows.min()) - 0.03),
        min(1.03, max(0.1, float(highs.max()) + 0.03)),
    )
    depth_axis.legend(loc="best")
    depth_axis.text(
        -0.15,
        1.03,
        "B",
        transform=depth_axis.transAxes,
        fontweight="bold",
        va="bottom",
    )

    metadata = {
        "requested_cells": requested_cell_count,
        "analyzed_cells": len(cells),
        "excluded_cells": paired_scope["excluded_cells"],
        "model_task_blocks": len(blocks),
        "pair_seeds": 5,
        "depth_positions": depths.tolist(),
        "grand_mean_matched": float(cells["matched"].mean()),
        "grand_mean_split": float(cells["split"].mean()),
        "bootstrap_draws": int(bootstrap_draws),
        "bootstrap_seed": int(bootstrap_seed),
    }
    return _save(figure, output_directory, stem, metadata)


def _epsilon_work(frame: pd.DataFrame) -> pd.DataFrame:
    model = _column(frame, "model", "model_key")
    task = _column(frame, "task", "task_key")
    method = _column(frame, "method")
    condition = _column(frame, "condition", "scoring_condition")
    epsilon = _column(frame, "epsilon", "eps")
    damage = _column(
        frame,
        "target_damage_C",
        "mean_target_damage_C",
        "C",
        "target_damage",
    )
    missing = [column for column in ("layer", "pair_seed") if column not in frame]
    if missing:
        raise FigureValidationError(f"epsilon rows lack columns: {missing!r}")
    work = pd.DataFrame(
        {
            "model": frame[model].astype(str),
            "task": frame[task].astype(str),
            "layer": frame["layer"].astype(int),
            "pair_seed": frame["pair_seed"].astype(int),
            "method": frame[method].astype(str).str.lower(),
            "condition": frame[condition].astype(str).str.lower(),
            "epsilon": pd.to_numeric(frame[epsilon], errors="coerce"),
            "damage": pd.to_numeric(frame[damage], errors="coerce"),
            "status": frame["status"].astype(str),
        }
    )
    if not np.isfinite(work["epsilon"].to_numpy(dtype=float)).all():
        raise FigureValidationError("epsilon contains non-finite values")
    observed_eps = tuple(sorted(float(value) for value in work["epsilon"].unique()))
    if observed_eps != LOCKED_EPSILONS:
        raise FigureValidationError(
            f"epsilon grid differs from the locked grid: {observed_eps!r}"
        )
    if set(work["method"]) != set(LOCKED_METHODS):
        raise FigureValidationError("epsilon rows require exactly FGSM and PGD")
    if set(work["condition"]) != set(LOCKED_CONDITIONS):
        raise FigureValidationError("epsilon rows require matched and split conditions")
    per_curve_epsilons = work.groupby(["method", "condition"])["epsilon"].agg(
        lambda values: tuple(sorted(float(value) for value in set(values)))
    )
    if (per_curve_epsilons != LOCKED_EPSILONS).any():
        raise FigureValidationError(
            "every method-condition epsilon grid must contain all ten locked values"
        )
    keys = ["model", "task", "layer", "pair_seed", "method", "epsilon"]
    if work.duplicated([*keys, "condition"]).any():
        raise FigureValidationError("epsilon rows contain duplicate scoring conditions")
    conditions = work.groupby(keys)["condition"].agg(frozenset)
    if (conditions != frozenset(LOCKED_CONDITIONS)).any():
        raise FigureValidationError("every epsilon unit requires matched and split rows")
    seeds = work.groupby(["model", "task", "layer", "method", "epsilon"])[
        "pair_seed"
    ].agg(lambda values: frozenset(int(value) for value in values))
    if (seeds != LOCKED_PAIR_SEEDS).any():
        raise FigureValidationError("every epsilon cell requires pair seeds 0 through 4")
    requested_curve_cells = work[
        ["model", "task", "layer", "method", "condition", "epsilon"]
    ].drop_duplicates()
    complete_mask = work.groupby(keys, sort=False, dropna=False)["status"].transform(
        lambda values: bool((values.astype(str) == "ok").all())
    )
    work = work.loc[complete_mask].copy()
    if work.empty:
        raise FigureValidationError("no complete epsilon pairs remain for plotting")
    _finite_unit_interval(work, ("damage",))
    analyzed_curve_cells = work[
        ["model", "task", "layer", "method", "condition", "epsilon"]
    ].drop_duplicates()
    analyzed_keys = set(analyzed_curve_cells.itertuples(index=False, name=None))
    excluded_curve_cells = [
        {
            "model": key[0],
            "task": key[1],
            "layer": int(key[2]),
            "method": key[3],
            "condition": key[4],
            "epsilon": float(key[5]),
        }
        for key in requested_curve_cells.itertuples(index=False, name=None)
        if tuple(key) not in analyzed_keys
    ]
    if work[["model", "task"]].drop_duplicates().shape[0] != 12:
        raise FigureValidationError("epsilon figure requires 12 model-task blocks")
    zero = work.loc[np.isclose(work["epsilon"], 0.0), "damage"].to_numpy(dtype=float)
    if zero.size == 0 or float(np.max(np.abs(zero))) > 1.0e-6:
        raise FigureValidationError("epsilon zero is not an exact no-op")
    work.attrs.update(
        {
            "requested_curve_cells": len(requested_curve_cells),
            "analyzed_curve_cells": len(analyzed_curve_cells),
            "excluded_curve_cells": excluded_curve_cells,
        }
    )
    return work


def _cluster_curve(
    rows: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    block_values = (
        rows.groupby(["model", "task"], sort=True)["damage"].mean().to_numpy(dtype=float)
    )
    if len(block_values) != 12:
        raise FigureValidationError("curve interval requires 12 model-task blocks")
    rng = np.random.default_rng(seed)
    selections = rng.integers(0, len(block_values), size=(draws, len(block_values)))
    estimates = block_values[selections].mean(axis=1)
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(block_values.mean()), float(low), float(high)


def _epsilon_positions() -> tuple[np.ndarray, list[str]]:
    positions = np.array(
        [-10.0 if epsilon == 0.0 else math.log2(epsilon) for epsilon in LOCKED_EPSILONS]
    )
    labels = ["0"] + [rf"$2^{{{int(math.log2(value))}}}$" for value in LOCKED_EPSILONS[1:]]
    return positions, labels


def create_epsilon_sweep_figure(
    rows_path: str | Path,
    output_directory: str | Path,
    *,
    bootstrap_draws: int = 10_000,
    bootstrap_seed: int = 20260830,
    stem: str = "fig_epsilon_sweep",
) -> FigureArtifacts:
    """Render FGSM/PGD damage curves and pair-level ceiling fractions."""

    if bootstrap_draws <= 0:
        raise FigureValidationError("bootstrap_draws must be positive")
    _publication_style()
    work = _epsilon_work(_saved_rows(rows_path))
    work_scope = dict(work.attrs)
    positions, tick_labels = _epsilon_positions()
    figure = plt.figure(figsize=(7.15, 4.35))
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=(3.1, 1.25),
        hspace=0.12,
        wspace=0.25,
    )
    top_axes = [figure.add_subplot(grid[0, index]) for index in range(2)]
    bottom_axes = [
        figure.add_subplot(grid[1, index], sharex=top_axes[index]) for index in range(2)
    ]
    style = {
        "matched": (OKABE_ITO["vermillion"], "o", "-"),
        "split": (OKABE_ITO["blue"], "s", "--"),
    }
    seed_offset = 0
    for method_index, method in enumerate(LOCKED_METHODS):
        top = top_axes[method_index]
        bottom = bottom_axes[method_index]
        method_rows = work.loc[work["method"] == method]
        for condition in LOCKED_CONDITIONS:
            means: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            ceiling_fractions: list[float] = []
            for epsilon in LOCKED_EPSILONS:
                subset = method_rows.loc[
                    (method_rows["condition"] == condition)
                    & np.isclose(method_rows["epsilon"], epsilon)
                ]
                mean, low, high = _cluster_curve(
                    subset,
                    draws=bootstrap_draws,
                    seed=bootstrap_seed + seed_offset,
                )
                seed_offset += 1
                means.append(mean)
                lows.append(low)
                highs.append(high)
                ceiling_fractions.append(
                    float(np.mean(np.isclose(subset["damage"], 1.0, atol=1.0e-12)))
                )
            means_array = np.asarray(means)
            color, marker, linestyle = style[condition]
            top.errorbar(
                positions,
                means_array,
                yerr=np.vstack((means_array - lows, np.asarray(highs) - means_array)),
                color=color,
                marker=marker,
                linestyle=linestyle,
                markersize=3.4,
                capsize=1.8,
                label=condition.capitalize(),
            )
            bottom.plot(
                positions,
                ceiling_fractions,
                color=color,
                marker=marker,
                linestyle=linestyle,
                markersize=3.0,
            )
        archived_position = math.log2(0.5)
        for axis in (top, bottom):
            axis.axvline(
                archived_position,
                color=OKABE_ITO["black"],
                linewidth=0.7,
                linestyle=":",
            )
            axis.set_xlim(positions.min() - 0.35, positions.max() + 0.25)
        top.set_ylim(-0.03, 1.03)
        bottom.set_ylim(-0.04, 1.04)
        top.tick_params(labelbottom=False)
        bottom.set_xticks(positions)
        bottom.set_xticklabels(tick_labels, rotation=45, ha="right")
        bottom.set_xlabel(r"Perturbation budget $\epsilon$ (log$_2$ spacing)")
        top.text(
            0.02,
            0.96,
            f"{chr(65 + method_index)}  {method.upper()}",
            transform=top.transAxes,
            fontweight="bold",
            va="top",
        )
        top.text(
            archived_position,
            0.04,
            " archived",
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=6.5,
        )
    top_axes[0].set_ylabel("Mean target damage, $C$")
    bottom_axes[0].set_ylabel("Pair fraction\nat $C=1$")
    top_axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.82))

    metadata = {
        "epsilons": list(LOCKED_EPSILONS),
        "methods": list(LOCKED_METHODS),
        "conditions": list(LOCKED_CONDITIONS),
        "model_task_blocks": 12,
        "pair_seeds": 5,
        **work_scope,
        "includes_ceiling_fraction": True,
        "bootstrap_draws": int(bootstrap_draws),
        "bootstrap_seed": int(bootstrap_seed),
    }
    return _save(figure, output_directory, stem, metadata)


def _candidate_construct_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ("edit_id", "evaluation_family", "label", "accuracy")
    missing = [column for column in required if column not in frame]
    if missing:
        raise FigureValidationError(f"construct rows lack columns: {missing!r}")
    if "edit_object" in frame:
        edit_objects = frame["edit_object"].astype(str).str.lower()
        candidates = frame.loc[edit_objects.isin(CANDIDATE_EDIT_OBJECTS)].copy()
    else:
        candidates = frame.loc[
            frame["edit_id"].astype(str).str.match(CANDIDATE_EDIT_ID)
        ].copy()
    if candidates.empty:
        raise FigureValidationError("construct rows contain no candidate edits")
    parsed = candidates["edit_id"].astype(str).str.extract(CANDIDATE_EDIT_ID)
    if parsed[["architecture", "seed"]].isna().any(axis=None):
        invalid_ids = candidates.loc[
            parsed["architecture"].isna(), "edit_id"
        ].astype(str).unique()
        raise FigureValidationError(
            f"candidate edit IDs do not match the archived or runner schema: {invalid_ids[:5]!r}"
        )
    if "architecture" not in candidates:
        candidates["architecture"] = parsed["architecture"].to_numpy()
    candidates["architecture"] = candidates["architecture"].astype(str).str.lower()
    parsed_architecture = parsed["architecture"].to_numpy()
    if not np.array_equal(candidates["architecture"].to_numpy(), parsed_architecture):
        raise FigureValidationError("candidate architecture disagrees with edit_id")
    if "candidate_seed" in candidates:
        candidate_seeds = pd.to_numeric(candidates["candidate_seed"], errors="coerce")
        parsed_seeds = pd.to_numeric(parsed["seed"], errors="coerce").to_numpy()
        if candidate_seeds.isna().any() or not np.array_equal(
            candidate_seeds.to_numpy(dtype=int), parsed_seeds.astype(int)
        ):
            raise FigureValidationError("candidate seed disagrees with edit_id")
    edit_architectures = candidates.groupby("edit_id")["architecture"].nunique()
    if (edit_architectures != 1).any():
        raise FigureValidationError("candidate architecture changes across evaluation rows")
    edit_table = candidates[["edit_id", "architecture"]].drop_duplicates()
    architecture_counts = edit_table.groupby("architecture")["edit_id"].nunique().to_dict()
    if architecture_counts != {"linear": 20, "mka": 20, "mlp": 20}:
        raise FigureValidationError(
            "construct figure requires 20 linear, 20 MLP, and 20 MKA candidate edits"
        )
    fixed = candidates.loc[
        (candidates["evaluation_family"].astype(str).str.lower() == "fixed")
        & (candidates["label"].astype(str).str.lower() == "target")
    ].copy()
    raw_column = _column(fixed, "C_raw", "target_damage_C_raw")
    oriented_column = _column(
        fixed,
        "C_orientation_calibrated",
        "C_orientation",
        "target_damage_C_orientation_calibrated",
    )
    fixed_values = (
        fixed.groupby(["edit_id", "architecture"], as_index=False)[
            [raw_column, oriented_column]
        ]
        .mean()
        .rename(columns={raw_column: "raw_damage", oriented_column: "oriented_damage"})
    )
    if len(fixed_values) != 60:
        raise FigureValidationError("every candidate edit requires one fixed-evaluator endpoint")
    family = candidates["evaluation_family"].astype(str).str.lower()
    fresh = candidates.loc[
        family.isin(("fresh_linear", "fresh_mlp"))
        & (candidates["label"].astype(str).str.lower() == "target")
    ].copy()
    fresh["evaluation_family"] = fresh["evaluation_family"].astype(str).str.lower()
    fresh_values = fresh.groupby(
        ["edit_id", "architecture", "evaluation_family"], as_index=False
    )["accuracy"].mean()
    coverage = fresh_values.groupby("edit_id")["evaluation_family"].agg(frozenset)
    if len(coverage) != 60 or (
        coverage != frozenset(("fresh_linear", "fresh_mlp"))
    ).any():
        raise FigureValidationError(
            "every candidate edit requires fresh linear and fresh MLP target endpoints"
        )
    _finite_unit_interval(fixed_values, ("raw_damage", "oriented_damage"))
    _finite_unit_interval(fresh_values, ("accuracy",))
    return fixed_values, fresh_values


def _distribution_panel(
    axis: plt.Axes,
    frame: pd.DataFrame,
    value_column: str,
    *,
    panel_label: str,
    ylabel: str,
) -> None:
    architectures = ("linear", "mlp", "mka")
    colors = (OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["green"])
    markers = ("o", "s", "^")
    for index, (architecture, color, marker) in enumerate(
        zip(architectures, colors, markers, strict=True)
    ):
        values = frame.loc[frame["architecture"] == architecture, value_column].to_numpy(
            dtype=float
        )
        offsets = np.linspace(-0.13, 0.13, len(values))
        axis.scatter(
            np.full(len(values), index, dtype=float) + offsets,
            values,
            color=color,
            marker=marker,
            s=15,
            edgecolors="white",
            linewidths=0.3,
            zorder=2,
        )
        median = float(np.median(values))
        axis.plot([index - 0.22, index + 0.22], [median, median], color="black", linewidth=1.4)
    axis.set_xticks(range(3))
    axis.set_xticklabels(("Linear", "MLP", "MKA"))
    axis.set_ylabel(ylabel)
    axis.set_ylim(-0.03, 1.03)
    axis.text(
        -0.12,
        1.03,
        panel_label,
        transform=axis.transAxes,
        fontweight="bold",
        va="bottom",
    )


def create_construct_check_figure(
    rows_path: str | Path,
    output_directory: str | Path,
    *,
    stem: str = "fig_orientation_redecodability",
) -> FigureArtifacts:
    """Render aligned candidate distributions for the three construct endpoints."""

    _publication_style()
    fixed, fresh = _candidate_construct_rows(_saved_rows(rows_path))
    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.75), sharey=True)
    _distribution_panel(
        axes[0],
        fixed,
        "raw_damage",
        panel_label="A  Raw fixed damage",
        ylabel="Endpoint value",
    )
    _distribution_panel(
        axes[1],
        fixed,
        "oriented_damage",
        panel_label="B  Calibrated damage",
        ylabel="Endpoint value",
    )
    architectures = ("linear", "mlp", "mka")
    colors = (OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["green"])
    for architecture_index, (architecture, color) in enumerate(
        zip(architectures, colors, strict=True)
    ):
        for family_index, (family, marker) in enumerate(
            (("fresh_linear", "o"), ("fresh_mlp", "^"))
        ):
            values = fresh.loc[
                (fresh["architecture"] == architecture)
                & (fresh["evaluation_family"] == family),
                "accuracy",
            ].to_numpy(dtype=float)
            center = architecture_index + (-0.14 if family_index == 0 else 0.14)
            offsets = np.linspace(-0.055, 0.055, len(values))
            axes[2].scatter(
                center + offsets,
                values,
                color=color,
                marker=marker,
                s=15,
                edgecolors="white",
                linewidths=0.3,
                zorder=2,
            )
            median = float(np.median(values))
            axes[2].plot(
                [center - 0.09, center + 0.09],
                [median, median],
                color="black",
                linewidth=1.2,
            )
    axes[2].set_xticks(range(3))
    axes[2].set_xticklabels(("Linear", "MLP", "MKA"))
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].text(
        -0.12,
        1.03,
        "C  Fresh target accuracy",
        transform=axes[2].transAxes,
        fontweight="bold",
        va="bottom",
    )
    axes[2].legend(
        handles=(
            Line2D([], [], color="black", marker="o", linestyle="None", label="Fresh linear"),
            Line2D([], [], color="black", marker="^", linestyle="None", label="Fresh MLP"),
        ),
        loc="lower right",
    )
    figure.subplots_adjust(wspace=0.15)
    metadata = {
        "candidate_edits": 60,
        "architectures": list(architectures),
        "endpoint_groups": [
            "raw_fixed_damage",
            "orientation_calibrated_damage",
            "fresh_decoder_accuracy",
        ],
    }
    return _save(figure, output_directory, stem, metadata)


def _row_artifact(run_directory: Path, stem: str) -> Path:
    for suffix in (".parquet", ".csv"):
        candidate = run_directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FigureValidationError(
        f"run directory lacks a saved {stem}.parquet or {stem}.csv artifact"
    )


def generate_revision_figures(
    run_directory: str | Path,
    *,
    output_directory: str | Path | None = None,
    bootstrap_draws: int = 10_000,
    bootstrap_seed: int = 20260830,
) -> dict[str, FigureArtifacts]:
    """Generate all locked figures from one immutable run directory."""

    run_path = Path(run_directory)
    destination = Path(output_directory) if output_directory else run_path / "figures"
    matched = create_expanded_matched_split_figure(
        _row_artifact(run_path, "matched_split_rows"),
        destination,
        bootstrap_draws=bootstrap_draws,
        bootstrap_seed=bootstrap_seed,
    )
    epsilon = create_epsilon_sweep_figure(
        _row_artifact(run_path, "epsilon_sweep_rows"),
        destination,
        bootstrap_draws=bootstrap_draws,
        bootstrap_seed=bootstrap_seed,
    )
    construct = create_construct_check_figure(
        _row_artifact(run_path, "construct_check_rows"),
        destination,
    )
    return {
        "matched_split": matched,
        "epsilon_sweep": epsilon,
        "construct_check": construct,
    }


__all__ = [
    "FigureArtifacts",
    "FigureValidationError",
    "create_construct_check_figure",
    "create_epsilon_sweep_figure",
    "create_expanded_matched_split_figure",
    "generate_revision_figures",
]
