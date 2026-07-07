"""
WS8 — Repaired v2 benchmark runner (one model x one task).

This is the v2 re-run runner. Unlike scripts/run_benchmark.py (which computes a
per-LAYER reliability that is identical across archs/seeds — Defect 1), this
runner ties reliability to EACH candidate probe's own extracted direction
(R_cand, WS5) and certifies it with independent bagged evaluator probes on a
disjoint fold. It also computes the full decomposition ladder (R_v1_max,
R_v1_mean, R_excl, R_cand) and the TVD variant (WS7) per method.

Locked design (PREREGISTRATION_v2.md):
  * 4-way disjoint split {candidate .40, evaluator .20, intervention .30,
    test .10} (D11), stratified by (zc, ze).
  * R_cand = the per-probe target; written to the record field "R" so
    scripts/prereg_v2_eval.py reads it directly. R_excl / R_v1_* ride along.
  * E=5 bagged evaluators (D1); EVAL_MIN_ACC=0.60 (D10); direction rank r=1,
    N_dir=4000 (D3); floor 0.55 (D4). Representations are z-scored per feature
    (candidate-fold statistics) before probe training so fixed-epoch probes
    train reliably across models/layers of differing scale.

Resume-safe (WS0.1), versioned output (results/benchmark_v2/, WS0.2), single-
pass cached extraction (WS0.4), provenance manifest — all reused from the v1
runner's contracts. Reliability construction is entirely composed from the
existing public APIs of src/{ws5_repaired,tvd,metrics,probes,interventions}.py.

Usage:
    python -m scripts.run_benchmark_v2 --config configs/pythia.yaml --task sva --k 20
    python -m scripts.run_benchmark_v2 --config configs/tiny.yaml --task sva --k 2   # smoke
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.pipeline import load_config
from src.extraction import load_model, select_layers, extract_all_layers, pick_device
from src.probes import ProbeTrainConfig, train_probe, probe_accuracy
from src.metrics import train_validation_probes, DECODABILITY_FLOOR
from src.gates import learnability_gate
from src.interventions import apply_alterrep, apply_fgsm, apply_pgd
from src.hessian import compute_hessian_spectrum
from src.repro import set_seed, hash_examples
from src.tasks import get_task
from src.tvd import compute_tvd_metrics
from src.ws5_repaired import (
    candidate_edit, train_independent_evaluators, repaired_reliability,
    concept_edits_once, EvaluatorQualityError, EVAL_SEED_BASE, EVAL_MIN_ACC,
)
from scripts.run_benchmark import (
    git_commit_hash, default_data_paths, load_completed_cells, make_probe, ARCHS,
)

SCHEMA_VERSION = 2

# Locked 4-way split fractions (PREREGISTRATION_v2.md D11).
SPLIT_FRACS = {"candidate": 0.40, "evaluator": 0.20, "intervention": 0.30, "test": 0.10}
N_DIR = 4000          # D3: fixed reference-set cap for direction extraction
DIRECTION_RANK = 1    # D3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--task", required=True, choices=["sva", "gender", "sst2"])
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--evaluators", type=int, default=5)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--layers", default=None, help="comma-separated layer override")
    p.add_argument("--data-paths", default=None)
    p.add_argument("--out-dir", default="results/benchmark_v2")
    p.add_argument("--cache-dir", default="cache/benchmark_v2")
    p.add_argument("--floor", type=float, default=DECODABILITY_FLOOR)
    p.add_argument("--no-tvd", action="store_true", help="skip TVD metrics (faster)")
    p.add_argument("--rank", type=int, default=DIRECTION_RANK)
    return p.parse_args()


def split_4way(examples, fracs, seed):
    """Deterministic, class-BALANCED, (zc,ze)-stratified 4-way split.

    Returns (candidate, evaluator, intervention, test) example lists, disjoint.

    The four (zc,ze) cells are first balanced to the minimum cell count (as the
    v1 runner's _split_balanced_4cell does) so BOTH zc and ze are ~50/50. This
    matters because the learnability gate and the metric floors use chance=0.5:
    on imbalanced data (Linzen SVA is zc 34/66, ze 32/69) a majority-class
    guesser scores at the imbalance rate, which spuriously (a) passes the 0.60
    floor without learning and (b) fails the shuffled-label control (a shuffled
    probe predicts the majority class, beating 0.5). Balancing makes 0.5 the
    correct baseline. Each balanced cell is then split by the fractions, so the
    four folds are class-balanced AND input-disjoint (given upstream dedup).
    """
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for ex in examples:
        buckets[(ex.zc, ex.ze)].append(ex)
    four = [(0, 0), (0, 1), (1, 0), (1, 1)]
    if all(k in buckets and buckets[k] for k in four):
        min_n = min(len(buckets[k]) for k in four)
        buckets = {k: (rng.sample(buckets[k], len(buckets[k])))[:min_n] for k in four}
    folds = {"candidate": [], "evaluator": [], "intervention": [], "test": []}
    order = ["candidate", "evaluator", "intervention", "test"]
    for key in sorted(buckets):
        grp = list(buckets[key])
        rng.shuffle(grp)
        n = len(grp)
        n_c = int(fracs["candidate"] * n)
        n_e = int(fracs["evaluator"] * n)
        n_i = int(fracs["intervention"] * n)
        folds["candidate"] += grp[:n_c]
        folds["evaluator"] += grp[n_c:n_c + n_e]
        folds["intervention"] += grp[n_c + n_e:n_c + n_e + n_i]
        folds["test"] += grp[n_c + n_e + n_i:]
    for k in order:
        rng.shuffle(folds[k])
    return folds["candidate"], folds["evaluator"], folds["intervention"], folds["test"]


def run_repaired_cell(arch, seed, *, X_cand, zc_cand, X_ref, X_inter, zc_inter,
                      ze_inter, X_test, zc_test, evaluators, concept_edits,
                      extra_edits, cfg, device, floor, rank, want_tvd):
    set_seed(seed)
    dim = X_cand.shape[1]
    probe = make_probe(arch, dim, hidden_dim=cfg["probes"]["hidden_dim"],
                       mka_lambda=cfg["mka"]["lambda_reg"], knn_k=cfg["mka"]["knn_k"],
                       seed=seed).to(device)
    train_cfg = ProbeTrainConfig(
        epochs=cfg["probes"]["epochs"], lr=cfg["probes"]["lr"],
        weight_decay=cfg["probes"]["weight_decay"], batch_size=cfg["probes"]["batch_size"],
    )
    train_probe(probe, X_cand, zc_cand, train_cfg, device)
    acc = probe_accuracy(probe, X_test, zc_test, device)

    spec = compute_hessian_spectrum(probe, X_cand, zc_cand, device,
                                    top_n=cfg["hessian"]["top_n"], bottom_n=0,
                                    max_iter=cfg["hessian"]["max_iter"], tol=cfg["hessian"]["tol"])
    A = spec.mean_align_top

    rr = repaired_reliability(
        probe, X_ref=X_ref, X_inter=X_inter, zc_inter=zc_inter, ze_inter=ze_inter,
        evaluators=evaluators, device=device, floor=floor, rank=rank,
        concept_edits=concept_edits, extra_edits=extra_edits,
    )

    tvd = {}
    if want_tvd:
        ev0 = evaluators[0].probes
        cand = candidate_edit(probe, X_ref, X_inter, device, r=rank)
        edits = {"cand": cand, **concept_edits, **extra_edits}
        for name, X_post in edits.items():
            if X_post is None:
                continue
            tvd[name] = compute_tvd_metrics(
                ev0, X_pre=X_inter, X_post=X_post, zc=zc_inter, ze=ze_inter,
                device=device, floor=floor,
            ).as_dict()

    return {
        "arch": arch, "seed": seed, "accuracy": acc,
        "A": A, "lambda_max": spec.lambda_max,
        # Per-probe target (R_cand) is written to "R" so prereg_v2_eval reads it.
        "R": rr["R_cand"], "R_method": "cand",
        "R_cand": rr["R_cand"], "R_excl": rr["R_excl"],
        "R_v1_max": rr["R_v1_max"], "R_v1_mean": rr["R_v1_mean"],
        "R_rep_per_method": rr["R_rep_per_method"],
        "R_rep_sd": rr["R_rep_sd"],
        "R_rep_n_eval_defined": rr["R_rep_n_eval_defined"],
        "degenerate_direction": rr["degenerate_direction"],
        "eval_acc_zc_oos": rr["eval_acc_zc_oos"],
        "eval_acc_ze_oos": rr["eval_acc_ze_oos"],
        "tvd": tvd,
        "num_params": probe.num_params(),
    }


def main():
    args = parse_args()
    cfg = load_config(args.config if Path(args.config).is_absolute()
                      else PROJECT_ROOT / args.config)
    if args.max_examples is not None:
        cfg["data"]["max_examples"] = args.max_examples
    if args.layers:
        cfg["extraction"]["layers"] = [int(x) for x in args.layers.split(",")]

    task = get_task(args.task)
    commit = git_commit_hash()
    print("=" * 72)
    print(f"  Benchmark v{SCHEMA_VERSION} REPAIRED: {cfg['model']['name']}  "
          f"task={task.name}  k={args.k}  E={args.evaluators}  commit={commit[:12]}")
    print("=" * 72)

    set_seed(cfg["output"]["seed"])
    data_paths = ([Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p
                   for p in args.data_paths.split(",")] if args.data_paths
                  else default_data_paths(task.name, cfg))
    examples = task.load(data_paths, max_examples=cfg["data"].get("max_examples"),
                         seed=cfg["data"]["seed"])
    print(f"[data] loaded {len(examples)} examples from {[str(p) for p in data_paths]}")

    # Deduplicate by sentence so the 4-way folds are INPUT-DISJOINT. Templated
    # data (Linzen SVA is ~14% repeated sentences) otherwise puts identical
    # inputs in different folds, leaking label information across the
    # candidate/evaluator/intervention boundary and inflating every metric — the
    # learnability gate's shuffled-label control catches it (shuffled acc ~0.63
    # instead of ~0.5). Keep the first occurrence of each sentence.
    seen: set = set()
    deduped = []
    for ex in examples:
        if ex.sentence in seen:
            continue
        seen.add(ex.sentence)
        deduped.append(ex)
    if len(deduped) < len(examples):
        print(f"[dedup] {len(examples)} -> {len(deduped)} unique sentences "
              f"({1 - len(deduped)/len(examples):.1%} duplicates removed)")
    examples = deduped

    cand_ex, eval_ex, inter_ex, test_ex = split_4way(
        examples, SPLIT_FRACS, seed=cfg["data"]["seed"])
    print(f"[split] candidate={len(cand_ex)} evaluator={len(eval_ex)} "
          f"intervention={len(inter_ex)} test={len(test_ex)}")

    device = pick_device()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[cfg["model"]["dtype"]]
    print(f"[model] loading {cfg['model']['name']} on {device}")
    bundle = load_model(cfg["model"]["name"], device=device, dtype=dtype,
                        trust_remote_code=cfg["model"].get("trust_remote_code", False))

    explicit = cfg["extraction"].get("layers")
    layers = list(explicit) if explicit else select_layers(
        bundle.n_layers, k=cfg["extraction"]["num_layers_to_probe"])
    print(f"[layers] {layers}")

    safe_name = cfg["model"]["name"].replace("/", "_")
    out_dir = (Path(args.out_dir) if Path(args.out_dir).is_absolute()
               else PROJECT_ROOT / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_name}_{task.name}.jsonl"
    manifest_path = out_dir / f"{safe_name}_{task.name}.manifest.json"
    done = load_completed_cells(out_path)
    total_cells = len(layers) * len(ARCHS) * args.k
    if done:
        print(f"[resume] {len(done)} completed cells ({total_cells - len(done)} remaining)")

    cache_dir = None
    if args.cache_dir:
        cache_dir = ((Path(args.cache_dir) if Path(args.cache_dir).is_absolute()
                      else PROJECT_ROOT / args.cache_dir) / f"{safe_name}_{task.name}")

    def layer_done(layer):
        return all((layer, a, 1000 + k) in done for a in ARCHS for k in range(args.k))
    pending_layers = [l for l in layers if not layer_done(l)]
    gate_layer = layers[len(layers) // 2]

    # Single-pass cached extraction for the 4 folds.
    probe_layers = sorted(set(pending_layers) | {gate_layer})
    reps = {}
    for tag, exs, need in (("cand", cand_ex, probe_layers), ("eval", eval_ex, pending_layers),
                           ("inter", inter_ex, pending_layers), ("test", test_ex, pending_layers)):
        if not need:
            reps[tag] = {}
            continue
        print(f"[extract] {tag} split, layers {need}")
        reps[tag] = extract_all_layers(
            bundle, exs, need, batch_size=cfg["extraction"]["batch_size"],
            max_length=cfg["extraction"]["max_length"], cache_dir=cache_dir, cache_tag=tag)

    # Standardize reps per layer (z-score using CANDIDATE-fold statistics) so
    # fixed-epoch probes train reliably across models/layers of differing
    # representation scale. Without this, probes on raw large-scale reps
    # undertrain and certifiers spuriously fail the quality gate (the gate
    # standardizes internally for exactly this reason, src/gates.py). A linear,
    # invertible transform — direction extraction and INLP-style erasure operate
    # consistently in this space. (v2 methodology; PREREGISTRATION_v2.md §2.)
    for layer in probe_layers:
        if layer not in reps.get("cand", {}):
            continue
        Xc = reps["cand"][layer]["X"]
        mu = Xc.mean(dim=0, keepdim=True)
        sd = Xc.std(dim=0, keepdim=True).clamp_min(1e-6)
        for tag in ("cand", "eval", "inter", "test"):
            if reps.get(tag) and layer in reps[tag]:
                reps[tag][layer]["X"] = (reps[tag][layer]["X"] - mu) / sd

    train_cfg = ProbeTrainConfig(
        epochs=cfg["probes"]["epochs"], lr=cfg["probes"]["lr"],
        weight_decay=cfg["probes"]["weight_decay"], batch_size=cfg["probes"]["batch_size"])

    # Learnability gate on the candidate fold at the middle layer.
    g = reps["cand"][gate_layer]
    gate = learnability_gate(g["X"], g["zc"], g["ze"], device, train_cfg,
                             zc_floor=task.zc_gate_floor, ze_floor=task.ze_gate_floor,
                             chance=task.chance_accuracy, layer=gate_layer)
    print(f"[gate] {gate.report()}")

    manifest = {"schema_version": SCHEMA_VERSION, "runs": []}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            manifest = {"schema_version": SCHEMA_VERSION, "runs": []}
    manifest["runs"].append({
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit, "config_file": str(args.config), "config": cfg,
        "task": task.name, "k": args.k, "evaluators": args.evaluators, "layers": layers,
        "floor": args.floor, "split_fracs": SPLIT_FRACS, "gate": gate.as_dict(),
        "data_paths": [str(p) for p in data_paths], "data_hash_all": hash_examples(examples),
        "n_examples": len(examples), "resumed_completed_cells": len(done),
        "repaired": True, "rank": args.rank, "tvd": (not args.no_tvd),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if not gate.passed:
        print("\n[FAIL] LEARNABILITY GATE FAILED — refusing to launch. See manifest:")
        print(f"   {manifest_path}")
        sys.exit(2)
    if not pending_layers:
        print(f"\n[done] Nothing to do: all {total_cells} cells complete. {out_path}")
        return

    inter_cfg_inlp = cfg["interventions"]["inlp"]["num_iters"]
    inter_cfg_rlace_rank = cfg["interventions"]["rlace"]["rank"]
    inter_cfg_rlace_steps = cfg["interventions"]["rlace"]["steps"]

    n_written, n_skipped = 0, 0
    with out_path.open("a") as f_out:
        for layer in pending_layers:
            X_cand, zc_cand = reps["cand"][layer]["X"], reps["cand"][layer]["zc"]
            X_eval = reps["eval"][layer]["X"]
            zc_eval, ze_eval = reps["eval"][layer]["zc"], reps["eval"][layer]["ze"]
            X_inter = reps["inter"][layer]["X"]
            zc_inter, ze_inter = reps["inter"][layer]["zc"], reps["inter"][layer]["ze"]
            X_test, zc_test = reps["test"][layer]["X"], reps["test"][layer]["zc"]
            X_ref = X_inter[:N_DIR]

            # Bagged independent certifiers (once per layer). A blind certifier
            # invalidates the layer (EvaluatorQualityError) — skip, don't crash.
            set_seed(cfg["output"]["seed"])
            try:
                evaluators = train_independent_evaluators(
                    X_eval, zc_eval, ze_eval, train_cfg, device,
                    n_evaluators=args.evaluators, min_acc=EVAL_MIN_ACC,
                    seed_base=EVAL_SEED_BASE + 101 * layer)
            except EvaluatorQualityError as e:
                print(f"[layer {layer}] SKIPPED — {e}")
                continue

            # Candidate-independent edits (once per layer).
            concept_edits = concept_edits_once(
                X_inter, zc_inter, device, inlp_iters=inter_cfg_inlp,
                rlace_rank=inter_cfg_rlace_rank, rlace_steps=inter_cfg_rlace_steps)
            set_seed(EVAL_SEED_BASE - 1)
            v1_vp = train_validation_probes(X_cand, zc_cand, reps["cand"][layer]["ze"],
                                            train_cfg, device, min_acc=0.0)
            extra_edits = {
                "AlterRep": apply_alterrep(X_inter, zc_inter, validation_probe=v1_vp.zc_probe,
                                           device=device, alpha=cfg["interventions"]["alterrep"]["alpha"]),
                "FGSM": apply_fgsm(X_inter, zc_inter, validation_probe=v1_vp.zc_probe,
                                   device=device, epsilon=cfg["interventions"]["fgsm"]["epsilon"]),
                "PGD": apply_pgd(X_inter, zc_inter, validation_probe=v1_vp.zc_probe,
                                 device=device, epsilon=cfg["interventions"]["pgd"]["epsilon"],
                                 steps=cfg["interventions"]["pgd"]["steps"],
                                 alpha=cfg["interventions"]["pgd"]["alpha"]),
            }
            print(f"[layer {layer}] evaluators={len(evaluators)} "
                  f"(zc_oos={[round(e.acc_zc_oos,3) for e in evaluators]})")

            for arch in ARCHS:
                for k in range(args.k):
                    seed = 1000 + k
                    if (layer, arch, seed) in done:
                        n_skipped += 1
                        continue
                    t1 = time.time()
                    row = run_repaired_cell(
                        arch, seed, X_cand=X_cand, zc_cand=zc_cand, X_ref=X_ref,
                        X_inter=X_inter, zc_inter=zc_inter, ze_inter=ze_inter,
                        X_test=X_test, zc_test=zc_test, evaluators=evaluators,
                        concept_edits=concept_edits, extra_edits=extra_edits,
                        cfg=cfg, device=device, floor=args.floor, rank=args.rank,
                        want_tvd=(not args.no_tvd))
                    row.update({"model": cfg["model"]["name"], "task": task.name,
                                "layer": layer, "wallclock_s": time.time() - t1,
                                "schema_version": SCHEMA_VERSION, "git_commit": commit})
                    f_out.write(json.dumps(row) + "\n")
                    f_out.flush()
                    n_written += 1
                    rc = "null" if row["R"] is None else f"{row['R']:.3f}"
                    re_str = "null" if row["R_excl"] is None else f"{row['R_excl']:.3f}"
                    print(f"  L{layer} {arch:6s} s={seed} acc={row['accuracy']:.3f} "
                          f"A={row['A']:.3f} Rcand={rc} Rexcl={re_str} "
                          f"[{row['wallclock_s']:.1f}s]")

    print(f"\n[done] Repaired benchmark complete: {out_path}")
    print(f"   wrote {n_written}, skipped {n_skipped}, total {len(done)+n_written}/{total_cells}")


if __name__ == "__main__":
    main()
