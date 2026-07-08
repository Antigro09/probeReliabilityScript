"""
WS4 — attacker / evaluator control on completeness.

WHY THIS EXISTS
---------------
Completeness (C) asks: "how much of the recoverable Zc signal did the
intervention remove?" In the benchmark it is measured with a validation
probe. But three of the five interventions — AlterRep, FGSM, PGD — are
*direction-based*: the edit is generated FROM a probe (the "attacker").
If the SAME probe both generates the edit and scores C (the "evaluator"),
completeness is inflated almost by construction: the edit pushes reps
across exactly the decision boundary the scorer reads off. That is a
probe-specific artifact, not evidence the concept was causally removed.

This module isolates that confound with a matched-vs-split control:

    MATCHED  — attacker probe == evaluator probe (same pair generates and
               scores). This is what the benchmark's per-method C measures.
    SPLIT    — attacker probe generates the edit; an INDEPENDENT evaluator
               probe (trained on a DISJOINT fold with a DISJOINT seed)
               scores C. This is the honest, transfer-tested completeness.

The reliability claim of interest is the GAP = matched_C - split_C. A large
positive gap for the direction-based methods, alongside a near-zero gap for
the probe-independent controls (INLP / RLACE, whose edits do not depend on
any single probe), is the WS4 signature that matched-probe completeness is
an over-estimate.

The projection interventions INLP and RLACE are reported as baselines:
their edits are population-level projections that use only (X, zc), so
matched vs split is not even definable for them — they are always scored
with the independent evaluator and their C is the reference the
direction-based split_C should be compared against.

DESIGN CONSTRAINTS (WS0.4)
--------------------------
This script CONSUMES cached clean representations only; it never loads a
model. Reps are the {X, zc, ze} .pt tensors written by
src.extraction.extract_all_layers during a benchmark run (see
scripts/run_benchmark.py). If a cache file is missing the script explains
how to produce it and exits — it will NOT silently fall back to model
loading.

The scientific core is attacker_evaluator_completeness(), a pure,
model-free function over representation tensors. It is unit-tested on
synthetic gaussian reps in tests/test_ws4.py (no GPU, no model, no cache).

Every threshold below that is an experimenter choice is marked
'lock in PREREGISTRATION_v2.md'.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.probes import ProbeTrainConfig
from src.metrics import (
    train_validation_probes,
    compute_intervention_metrics,
    DECODABILITY_FLOOR,
)
from src.interventions import (
    InterventionConfig,
    apply_alterrep,
    apply_fgsm,
    apply_pgd,
    apply_inlp,
    apply_rlace,
)
from src.repro import set_seed
from src.extraction import pick_device


# ---------------------------------------------------------------------------
# Experimenter choices — lock in PREREGISTRATION_v2.md before the v2 run.
# ---------------------------------------------------------------------------

# Number of independently-seeded validation-probe PAIRS per fold. C is
# aggregated across these pairs and the CI is bootstrapped over them, so this
# controls how much probe-init/training noise the CI reflects.
DEFAULT_M = 5  # lock in PREREGISTRATION_v2.md

# Bootstrap resamples for the 95% CI over the M pairs.
DEFAULT_BOOTSTRAP = 2000  # lock in PREREGISTRATION_v2.md

# CI coverage. 95% two-sided -> 2.5 / 97.5 percentiles.
CI_ALPHA = 0.05  # lock in PREREGISTRATION_v2.md

# Direction-based (probe-generated) attacks: matched vs split is meaningful.
DIRECTION_METHODS = ("AlterRep", "FGSM", "PGD")  # lock in PREREGISTRATION_v2.md

# Projection controls: probe-independent edits, reported as baselines only.
CONTROL_METHODS = ("INLP", "RLACE")  # lock in PREREGISTRATION_v2.md

# Seed base offsets so attacker- and evaluator-fold probe seeds never collide
# with each other or with the fold-splitting seed.
_ATTACKER_SEED_BASE = 10_000
_EVALUATOR_SEED_BASE = 20_000

# The registered WS4 subset: all six benchmarked models, MIDDLE probed layer,
# on the two non-gender tasks. lock in PREREGISTRATION_v2.md.
DEFAULT_MODELS = (
    "google-bert/bert-base-uncased",
    "google/gemma-2-2b",
    "openai-community/gpt2",
    "meta-llama/Llama-3.2-3B",
    "EleutherAI/pythia-160m",
    "Qwen/Qwen2.5-1.5B",
)  # lock in PREREGISTRATION_v2.md
DEFAULT_TASKS = ("sva", "sst2")  # lock in PREREGISTRATION_v2.md

# MIDDLE-layer convention: the benchmark's gate layer is the middle of the
# select_layers() schedule (run_benchmark.py: gate_layer =
# layers[len(layers) // 2]). WS4 mirrors that by choosing the MIDDLE of the
# probed layers actually present in the cache for each cell (see
# _find_cache_file), so it lands on the same layer without re-deriving depth.


# ---------------------------------------------------------------------------
# Fold splitting (deterministic, seeded)
# ---------------------------------------------------------------------------

def _three_way_split(
    n: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Deterministically partition indices [0, n) into three disjoint folds:
        attacker-train, evaluator-train, intervention.

    A seeded torch.Generator drives a permutation so the split is exactly
    reproducible from (n, seed) alone — no global RNG state dependence.
    The intervention fold is the largest (edits/scoring happen there); the
    two probe-training folds are equal-sized halves of the remainder.
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    # Roughly 1/3 attacker, 1/3 evaluator, 1/3 intervention. Integer split
    # keeps folds disjoint and covers all n indices.
    a_end = n // 3
    e_end = 2 * (n // 3)
    attacker_idx = perm[:a_end]
    evaluator_idx = perm[a_end:e_end]
    inter_idx = perm[e_end:]
    return attacker_idx, evaluator_idx, inter_idx


def _train_probe_pairs(
    X: torch.Tensor,
    zc: torch.Tensor,
    ze: torch.Tensor,
    cfg: ProbeTrainConfig,
    device: torch.device,
    m: int,
    seed_base: int,
):
    """
    Train M independently-seeded validation-probe PAIRS on one fold.

    train_validation_probes takes no seed argument; per its contract we seed
    the global RNG via src.repro.set_seed immediately before each call so the
    M pairs differ by probe init and mini-batch order. Seeds are drawn from a
    fixed base so attacker- and evaluator-fold seeds are disjoint.
    """
    pairs = []
    for i in range(m):
        set_seed(seed_base + i)
        vp = train_validation_probes(X, zc, ze, cfg, device, min_acc=0.0)
        pairs.append(vp)
    return pairs


# ---------------------------------------------------------------------------
# Aggregation + bootstrap
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    values: list[float], rng: np.random.Generator, n_boot: int
) -> tuple[float | None, float | None]:
    """
    Percentile bootstrap 95% CI of the MEAN over `values` (one value per
    probe pair). Returns (low, high). None when there is nothing to resample.

    A single value yields a degenerate (v, v) interval, which is the honest
    statement that with one pair the sampling spread is unestimable.
    """
    vals = np.asarray([v for v in values if v is not None], dtype=float)
    if vals.size == 0:
        return None, None
    if vals.size == 1:
        return float(vals[0]), float(vals[0])
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    boot_means = vals[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * (CI_ALPHA / 2)))
    hi = float(np.percentile(boot_means, 100 * (1 - CI_ALPHA / 2)))
    return lo, hi


def _mean_or_none(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Core pure function (model-free, unit-testable on synthetic reps)
# ---------------------------------------------------------------------------

def attacker_evaluator_completeness(
    X: torch.Tensor,
    zc: torch.Tensor,
    ze: torch.Tensor,
    device: torch.device,
    *,
    M: int = DEFAULT_M,
    cfg: ProbeTrainConfig,
    inter_cfg: InterventionConfig,
    floor: float = DECODABILITY_FLOOR,
    methods: tuple[str, ...] = DIRECTION_METHODS,
    controls: tuple[str, ...] = CONTROL_METHODS,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
) -> dict:
    """
    Quantify the attacker/evaluator confound on completeness (C).

    Pipeline
    --------
    1. Split (X, zc, ze) into three disjoint, seeded folds:
       attacker-train, evaluator-train, intervention.
    2. Train M independently-seeded validation-probe PAIRS on the attacker
       fold and M on the evaluator fold (disjoint seed schedules).
    3. For each direction-based method in `methods`, generate X_post on the
       intervention fold from the i-th ATTACKER probe (validation_probe =
       attacker_pairs[i].zc_probe), then score C two ways:
         MATCHED — score with the SAME attacker pair.
         SPLIT   — score with the i-th INDEPENDENT evaluator pair.
    4. For each control in `controls`, apply the probe-independent edit
       (INLP / RLACE need only X, zc) and score C with the evaluator pairs.
       Controls are baselines; there is no matched/split distinction.
    5. Aggregate C across the M pairs (skipping None below the floor) and
       bootstrap a 95% CI over the pairs with numpy default_rng(seed).

    Returns
    -------
    dict with:
        "methods": {
            method: {
                "matched_C_mean", "matched_C_ci": [lo, hi], "matched_n",
                "split_C_mean",   "split_C_ci":   [lo, hi], "split_n",
                "gap"  (matched_C_mean - split_C_mean, None if either None),
                "matched_C_pairs", "split_C_pairs"  (raw per-pair C, nullable),
            } for method in methods
        },
        "controls": {
            control: {
                "C_mean", "C_ci": [lo, hi], "n", "C_pairs",
            } for control in controls
        },
        "M", "bootstrap", "seed", "floor", "n_attacker", "n_evaluator",
        "n_intervention".

    Notes
    -----
    C is compute_intervention_metrics(...).completeness — floored and
    nullable. None values (a pair whose acc_zc_pre fell below `floor`) are
    dropped from the mean/CI and counted in *_n so a degenerate fold is
    visible rather than silently biasing the estimate.
    """
    n = X.shape[0]
    attacker_idx, evaluator_idx, inter_idx = _three_way_split(n, seed)

    X_att, zc_att, ze_att = X[attacker_idx], zc[attacker_idx], ze[attacker_idx]
    X_eval, zc_eval, ze_eval = (
        X[evaluator_idx], zc[evaluator_idx], ze[evaluator_idx],
    )
    X_int, zc_int, ze_int = X[inter_idx], zc[inter_idx], ze[inter_idx]

    # Two independent banks of M probe pairs, disjoint seed schedules.
    attacker_pairs = _train_probe_pairs(
        X_att, zc_att, ze_att, cfg, device, M, _ATTACKER_SEED_BASE
    )
    evaluator_pairs = _train_probe_pairs(
        X_eval, zc_eval, ze_eval, cfg, device, M, _EVALUATOR_SEED_BASE
    )

    rng = np.random.default_rng(seed)

    def _apply_direction(method: str, val_probe) -> torch.Tensor:
        """Generate X_post on the intervention fold from a validation probe."""
        if method == "AlterRep":
            return apply_alterrep(
                X_int, zc_int, validation_probe=val_probe, device=device,
                alpha=inter_cfg.alterrep_alpha,
            )
        if method == "FGSM":
            return apply_fgsm(
                X_int, zc_int, validation_probe=val_probe, device=device,
                epsilon=inter_cfg.fgsm_eps,
            )
        if method == "PGD":
            return apply_pgd(
                X_int, zc_int, validation_probe=val_probe, device=device,
                epsilon=inter_cfg.pgd_eps, steps=inter_cfg.pgd_steps,
                alpha=inter_cfg.pgd_alpha,
            )
        raise ValueError(f"Unknown direction-based method: {method!r}")

    def _completeness(val_probes, X_post: torch.Tensor) -> float | None:
        m = compute_intervention_metrics(
            val_probes, X_pre=X_int, X_post=X_post,
            zc=zc_int, ze=ze_int, device=device, floor=floor,
        )
        return m.completeness  # floored, nullable

    methods_out: dict[str, dict] = {}
    for method in methods:
        matched_C: list[float | None] = []
        split_C: list[float | None] = []
        for i in range(M):
            attacker = attacker_pairs[i]
            evaluator = evaluator_pairs[i]
            # The attacker's Zc probe both generates the edit and (in the
            # matched condition) certifies it.
            X_post = _apply_direction(method, attacker.zc_probe)
            matched_C.append(_completeness(attacker, X_post))
            split_C.append(_completeness(evaluator, X_post))

        matched_mean = _mean_or_none(matched_C)
        split_mean = _mean_or_none(split_C)
        matched_lo, matched_hi = _bootstrap_ci(matched_C, rng, bootstrap)
        split_lo, split_hi = _bootstrap_ci(split_C, rng, bootstrap)
        gap = (
            None if (matched_mean is None or split_mean is None)
            else matched_mean - split_mean
        )
        methods_out[method] = {
            "matched_C_mean": matched_mean,
            "matched_C_ci": [matched_lo, matched_hi],
            "matched_n": sum(1 for v in matched_C if v is not None),
            "split_C_mean": split_mean,
            "split_C_ci": [split_lo, split_hi],
            "split_n": sum(1 for v in split_C if v is not None),
            "gap": gap,
            "matched_C_pairs": matched_C,
            "split_C_pairs": split_C,
        }

    controls_out: dict[str, dict] = {}
    for control in controls:
        # Probe-independent edit: computed ONCE (it does not depend on any
        # probe), then scored with each evaluator pair to reflect scoring
        # noise across the M evaluator probes.
        if control == "INLP":
            X_post = apply_inlp(
                X_int, zc_int, device=device, num_iters=inter_cfg.inlp_iters,
            )
        elif control == "RLACE":
            X_post = apply_rlace(
                X_int, zc_int, device=device, rank=inter_cfg.rlace_rank,
                steps=inter_cfg.rlace_steps,
            )
        else:
            raise ValueError(f"Unknown control method: {control!r}")
        C_vals = [_completeness(evaluator_pairs[i], X_post) for i in range(M)]
        c_lo, c_hi = _bootstrap_ci(C_vals, rng, bootstrap)
        controls_out[control] = {
            "C_mean": _mean_or_none(C_vals),
            "C_ci": [c_lo, c_hi],
            "n": sum(1 for v in C_vals if v is not None),
            "C_pairs": C_vals,
        }

    return {
        "methods": methods_out,
        "controls": controls_out,
        "M": M,
        "bootstrap": bootstrap,
        "seed": seed,
        "floor": floor,
        "n_attacker": int(attacker_idx.numel()),
        "n_evaluator": int(evaluator_idx.numel()),
        "n_intervention": int(inter_idx.numel()),
    }


# ---------------------------------------------------------------------------
# CLI: consume cached clean reps, run per (model, task), write JSON
# ---------------------------------------------------------------------------

def _find_cache_file(
    cache_root: Path, safe_model: str, task: str, cache_tag: str,
    layer: int | None,
) -> tuple[Path | None, list[int]]:
    """
    Locate the cached reps .pt for (model, task, tag). The datahash suffix is
    unknown ahead of time, so we glob. Returns (path_or_None, layers_present).

    Cache layout mirrors run_benchmark.py:
        <cache_root>/<safe_model>_<task>/<safe_model>_<tag>_L<layer>_<hash>.pt
    """
    cell_dir = cache_root / f"{safe_model}_{task}"
    pattern = f"{safe_model}_{cache_tag}_L*.pt"
    matches = sorted(cell_dir.glob(pattern))
    layers_present: list[int] = []
    by_layer: dict[int, Path] = {}
    for p in matches:
        # filename: <safe_model>_<tag>_L<layer>_<hash>.pt
        # The layer token is "L" followed by digits; the trailing datahash is
        # lowercase hex so it never starts with "L", and taking the LAST such
        # token is robust even when a model-name segment starts with "L"
        # (e.g. "Llama-3.2-3B").
        stem = p.stem
        layer_tokens = [
            t for t in stem.split("_")
            if len(t) > 1 and t[0] == "L" and t[1:].isdigit()
        ]
        if not layer_tokens:
            continue
        lyr = int(layer_tokens[-1][1:])
        layers_present.append(lyr)
        by_layer.setdefault(lyr, p)
    layers_present = sorted(set(layers_present))
    if not layers_present:
        return None, []
    if layer is None:
        # Mirror the benchmark's gate-layer convention: MIDDLE of the sorted
        # probed layers present in the cache.
        chosen = layers_present[len(layers_present) // 2]
    else:
        chosen = layer
    return by_layer.get(chosen), layers_present


def _load_clean_reps(path: Path):
    """Load the {X, zc, ze} cache dict written by extract_all_layers."""
    d = torch.load(path, weights_only=True)
    return d["X"], d["zc"], d["ze"]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--cache-dir", default="cache/benchmark_v2",
        help="Root of the benchmark rep cache (default matches "
             "run_benchmark.py). Per-cell dirs are <root>/<safe_model>_<task>/.",
    )
    p.add_argument(
        "--cache-tag", default="inter", choices=["probe", "inter", "test"],
        help="Which cached split to draw clean reps from. Default 'inter' — "
             "the benchmark's intervention split. The pure function re-splits "
             "into its own disjoint folds regardless, so this only selects the "
             "population of clean reps.",
    )
    p.add_argument(
        "--models", default=None,
        help="Comma-separated model names; default is the registered WS4 "
             "subset (all six benchmarked models).",
    )
    p.add_argument(
        "--tasks", default=None,
        help="Comma-separated tasks; default 'sva,sst2' (registered subset).",
    )
    p.add_argument(
        "--layer", type=int, default=None,
        help="Explicit layer index override. Default: the MIDDLE probed layer "
             "present in the cache (the benchmark gate layer).",
    )
    p.add_argument(
        "--M", type=int, default=DEFAULT_M,
        help=f"Probe pairs per fold (default {DEFAULT_M}; "
             "lock in PREREGISTRATION_v2.md).",
    )
    p.add_argument(
        "--bootstrap", type=int, default=DEFAULT_BOOTSTRAP,
        help=f"Bootstrap resamples (default {DEFAULT_BOOTSTRAP}; "
             "lock in PREREGISTRATION_v2.md).",
    )
    p.add_argument(
        "--floor", type=float, default=DECODABILITY_FLOOR,
        help="Decodability floor for C (see src/metrics.py; "
             "lock in PREREGISTRATION_v2.md).",
    )
    p.add_argument("--seed", type=int, default=0,
                   help="Master seed for fold split, probe seeding, bootstrap.")
    p.add_argument(
        "--out", default="results/ws4/attacker_evaluator.json",
        help="Output JSON path.",
    )
    return p.parse_args()


def _default_probe_cfg() -> ProbeTrainConfig:
    """
    Validation-probe training config for WS4.

    Matches the benchmark's default ProbeTrainConfig (epochs=50, lr=1e-3,
    weight_decay=0.01, batch_size=256 — see src/probes.py). WS4 draws no probe
    hyperparameters from any single model config: the whole point is a
    consistent probe across models, so we use the shared default.
    lock in PREREGISTRATION_v2.md.
    """
    return ProbeTrainConfig()


def _default_inter_cfg() -> InterventionConfig:
    """Intervention hyperparameters — the benchmark defaults (src/interventions.py).
    lock in PREREGISTRATION_v2.md."""
    return InterventionConfig()


def main():
    args = parse_args()
    device = pick_device()

    cache_root = (
        Path(args.cache_dir) if Path(args.cache_dir).is_absolute()
        else PROJECT_ROOT / args.cache_dir
    )
    models = (
        [m.strip() for m in args.models.split(",")] if args.models
        else list(DEFAULT_MODELS)
    )
    tasks = (
        [t.strip() for t in args.tasks.split(",")] if args.tasks
        else list(DEFAULT_TASKS)
    )

    out_path = (
        Path(args.out) if Path(args.out).is_absolute() else PROJECT_ROOT / args.out
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = _default_probe_cfg()
    inter_cfg = _default_inter_cfg()

    print("=" * 70)
    print("  WS4 attacker/evaluator completeness control")
    print(f"  cache={cache_root}  tag={args.cache_tag}  device={device}")
    print(f"  M={args.M}  bootstrap={args.bootstrap}  floor={args.floor}")
    print("=" * 70)

    results: dict[str, dict] = {}
    missing: list[str] = []

    for model in models:
        safe_model = model.replace("/", "_")
        for task in tasks:
            cell_key = f"{safe_model}|{task}"
            path, layers_present = _find_cache_file(
                cache_root, safe_model, task, args.cache_tag, args.layer,
            )
            if path is None:
                cell_dir = cache_root / f"{safe_model}_{task}"
                if layers_present and args.layer is not None:
                    # The cell HAS reps, just not the explicitly requested layer.
                    print(f"  ✗ {cell_key}: requested layer L{args.layer} not in "
                          f"cache (layers present: {layers_present}) in {cell_dir}")
                else:
                    print(f"  ✗ {cell_key}: no cached '{args.cache_tag}' reps in "
                          f"{cell_dir}")
                missing.append(cell_key)
                continue
            print(f"  • {cell_key}: loading {path.name} "
                  f"(layers present: {layers_present})")
            X, zc, ze = _load_clean_reps(path)

            res = attacker_evaluator_completeness(
                X, zc, ze, device,
                M=args.M, cfg=cfg, inter_cfg=inter_cfg, floor=args.floor,
                methods=DIRECTION_METHODS, controls=CONTROL_METHODS,
                bootstrap=args.bootstrap, seed=args.seed,
            )
            res["model"] = model
            res["task"] = task
            res["cache_file"] = str(path)
            res["layers_present"] = layers_present
            results[cell_key] = res

            for method, mo in res["methods"].items():
                def _f(v):
                    return "null" if v is None else f"{v:.3f}"
                print(f"      {method:8s} matched_C={_f(mo['matched_C_mean'])} "
                      f"split_C={_f(mo['split_C_mean'])} gap={_f(mo['gap'])}")

    payload = {
        "schema": "ws4_attacker_evaluator",
        "params": {
            "M": args.M,
            "bootstrap": args.bootstrap,
            "floor": args.floor,
            "seed": args.seed,
            "cache_tag": args.cache_tag,
            "direction_methods": list(DIRECTION_METHODS),
            "control_methods": list(CONTROL_METHODS),
            "ci_alpha": CI_ALPHA,
        },
        "cells": results,
        "missing_cells": missing,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print("=" * 70)
    print(f"  wrote {len(results)} cells to {out_path}")

    if missing:
        print()
        print("  Missing cached reps for: " + ", ".join(missing))
        print("  WS4 consumes cached representations only (WS0.4) and will not")
        print("  load models. Produce the cache with a benchmark run, e.g.:")
        print()
        print("     python -m scripts.run_benchmark --config configs/<model>.yaml \\")
        print(f"         --task <task> --cache-dir {args.cache_dir}")
        print()
        print("  then re-run this script with the same --cache-dir.")
        # A partial run still writes what it could; missing cells are a data
        # availability issue, not a code failure. Exit non-zero so an
        # orchestrator notices the subset is incomplete.
        sys.exit(3)


if __name__ == "__main__":
    main()
