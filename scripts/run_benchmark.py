"""
K-seed benchmark runner for the alignment-as-predictor experiment.

For each (model, layer, task, architecture) combination, train K probes
with different random seeds. For each probe, record:
    - test accuracy
    - mean directional alignment with top-20 Hessian eigenvectors (the predictor A)
    - max reliability across the five intervention methods (the target R)

This produces the dataset used by predictor_eval.py to compute Spearman
correlation and rank-1 hit rate against the pre-registered thresholds.

Usage:
    # Run the full benchmark (slow, days of compute):
    python -m scripts.run_benchmark --config configs/pythia.yaml --task sva --k 20

    # Quick smoke test (single layer, k=2):
    python -m scripts.run_benchmark --config configs/tiny.yaml --task sva --k 2

Output:
    results/benchmark/<model>_<task>.jsonl
    one line per probe = one (model, layer, task, architecture, seed) cell.

The JSONL format keeps individual seeds visible so the predictor evaluator
can compute median-over-seeds aggregations honestly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.pipeline import load_config
from src.extraction import (
    load_model, select_layers, extract_layer_reps, pick_device,
)
from src.probes import (
    LinearProbe, MLPProbe, MKAProbe,
    ProbeTrainConfig, train_probe, probe_accuracy,
)
from src.metrics import train_validation_probes, compute_intervention_metrics
from src.interventions import InterventionConfig, run_all_interventions
from src.hessian import compute_hessian_spectrum
from src.repro import set_seed
from src.tasks import get_task


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True,
                   help="Model config YAML (e.g., configs/pythia.yaml)")
    p.add_argument("--task", required=True, choices=["sva", "gender", "sst2"],
                   help="Task name")
    p.add_argument("--k", type=int, default=20,
                   help="Number of seeds per (architecture, layer) cell")
    p.add_argument("--max-examples", type=int, default=None,
                   help="Override config's max_examples")
    p.add_argument("--layers", default=None,
                   help="Comma-separated layers; defaults to config")
    p.add_argument("--data-paths", default=None,
                   help="Comma-separated paths for the task data; "
                        "defaults to config['data']['paths']")
    return p.parse_args()


def make_probe(arch: str, dim: int, hidden_dim: int,
               mka_lambda: float, knn_k: int, seed: int):
    """Build a probe of the given architecture with seed-controlled init."""
    g = torch.Generator().manual_seed(seed)
    if arch == "linear":
        probe = LinearProbe(dim)
    elif arch == "mlp":
        probe = MLPProbe(dim, hidden_dim=hidden_dim)
    elif arch == "mka":
        probe = MKAProbe(dim, hidden_dim=hidden_dim,
                         mka_lambda=mka_lambda, knn_k=knn_k)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    # Re-init with seeded generator so different seeds get different init.
    for p in probe.parameters():
        if p.dim() >= 2:
            torch.nn.init.kaiming_uniform_(p, a=5**0.5, generator=g)
        else:
            torch.nn.init.zeros_(p)
    return probe


def run_one_cell(
    arch: str,
    seed: int,
    X_probe, zc_probe, ze_probe,
    X_inter, zc_inter, ze_inter,
    X_test, zc_test,
    val_probes,
    interventions_dict,  # pre-computed once per layer (don't recompute per probe)
    cfg: dict,
    device: torch.device,
) -> dict:
    """Train one probe at one (arch, seed), compute A and R."""
    # Per-seed determinism. Shuffles in train_probe will pick up this seed.
    set_seed(seed)

    dim = X_probe.shape[1]
    probe = make_probe(
        arch, dim,
        hidden_dim=cfg["probes"]["hidden_dim"],
        mka_lambda=cfg["mka"]["lambda_reg"],
        knn_k=cfg["mka"]["knn_k"],
        seed=seed,
    ).to(device)

    train_cfg = ProbeTrainConfig(
        epochs=cfg["probes"]["epochs"],
        lr=cfg["probes"]["lr"],
        weight_decay=cfg["probes"]["weight_decay"],
        batch_size=cfg["probes"]["batch_size"],
    )
    train_probe(probe, X_probe, zc_probe, train_cfg, device)
    acc = probe_accuracy(probe, X_test, zc_test, device)

    # ---- Predictor A: directional alignment with top-20 eigvecs ----
    spec = compute_hessian_spectrum(
        probe, X_probe, zc_probe, device,
        top_n=cfg["hessian"]["top_n"], bottom_n=0,  # we only need top
        max_iter=cfg["hessian"]["max_iter"], tol=cfg["hessian"]["tol"],
    )
    A = spec.mean_align_top  # mean |cos| with top-N eigenvectors
    lambda_max = spec.lambda_max

    # ---- Target R: max reliability across the 5 interventions ----
    # Important: re-train validation probes WOULD leak. We use the
    # pre-computed val_probes (trained on un-intervened reps per-layer).
    best_R = -1.0
    best_method = None
    per_method = {}
    for method, X_post in interventions_dict.items():
        m = compute_intervention_metrics(
            val_probes, X_pre=X_inter, X_post=X_post,
            zc=zc_inter, ze=ze_inter, device=device,
        )
        per_method[method] = {
            "C": m.completeness, "S": m.selectivity, "R": m.reliability,
        }
        if m.reliability > best_R:
            best_R = m.reliability
            best_method = method

    return {
        "arch": arch,
        "seed": seed,
        "accuracy": acc,
        "A": A,
        "lambda_max": lambda_max,
        "R": best_R,
        "R_method": best_method,
        "per_method": per_method,
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
    print("=" * 70)
    print(f"  Benchmark: {cfg['model']['name']}  task={task.name}  k={args.k}")
    print("=" * 70)

    set_seed(cfg["output"]["seed"])

    # Data paths -- use task-specific overrides if provided, else config defaults.
    if args.data_paths:
        data_paths = [Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p
                      for p in args.data_paths.split(",")]
    else:
        data_paths = [PROJECT_ROOT / p for p in cfg["data"]["paths"]]

    print(f"[data] task={task.name}  loading from {[str(p) for p in data_paths]}")
    examples = task.load(data_paths,
                          max_examples=cfg["data"].get("max_examples"),
                          seed=cfg["data"]["seed"])
    print(f"[data] loaded {len(examples)} examples")
    train_ex, inter_ex, test_ex = task.split(
        examples, val_frac=cfg["data"]["val_frac"],
        inter_frac=cfg["data"]["inter_frac"], seed=cfg["data"]["seed"],
    )
    print(f"[data] probe-train={len(train_ex)} inter={len(inter_ex)} test={len(test_ex)}")

    # Model
    device = pick_device()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[cfg["model"]["dtype"]]
    print(f"[model] loading {cfg['model']['name']} on {device}")
    bundle = load_model(
        cfg["model"]["name"], device=device, dtype=dtype,
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    )

    # Layers
    explicit = cfg["extraction"].get("layers")
    layers = list(explicit) if explicit else select_layers(
        bundle.n_layers, k=cfg["extraction"]["num_layers_to_probe"]
    )
    print(f"[layers] {layers}")

    # Output
    safe_name = cfg["model"]["name"].replace("/", "_")
    out_dir = PROJECT_ROOT / "results" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_name}_{task.name}.jsonl"
    print(f"[output] {out_path}")

    archs = ["linear", "mlp", "mka"]
    inter_cfg = InterventionConfig(
        inlp_iters=cfg["interventions"]["inlp"]["num_iters"],
        rlace_rank=cfg["interventions"]["rlace"]["rank"],
        rlace_steps=cfg["interventions"]["rlace"]["steps"],
        alterrep_alpha=cfg["interventions"]["alterrep"]["alpha"],
        fgsm_eps=cfg["interventions"]["fgsm"]["epsilon"],
        pgd_eps=cfg["interventions"]["pgd"]["epsilon"],
        pgd_steps=cfg["interventions"]["pgd"]["steps"],
        pgd_alpha=cfg["interventions"]["pgd"]["alpha"],
    )

    # Open results file in append mode so we can resume after interruption.
    f_out = out_path.open("a")

    for layer in layers:
        print(f"\n[layer {layer}] extracting representations...")
        t0 = time.time()
        X_probe, zc_probe, ze_probe = extract_layer_reps(
            bundle, train_ex, layer_idx=layer,
            batch_size=cfg["extraction"]["batch_size"],
            max_length=cfg["extraction"]["max_length"],
        )
        X_inter, zc_inter, ze_inter = extract_layer_reps(
            bundle, inter_ex, layer_idx=layer,
            batch_size=cfg["extraction"]["batch_size"],
            max_length=cfg["extraction"]["max_length"],
            validate=False,
        )
        X_test, zc_test, _ = extract_layer_reps(
            bundle, test_ex, layer_idx=layer,
            batch_size=cfg["extraction"]["batch_size"],
            max_length=cfg["extraction"]["max_length"],
            validate=False,
        )
        print(f"[layer {layer}] extraction: {time.time() - t0:.1f}s")

        # Validation probes & interventions are computed ONCE per layer.
        # They depend only on the representations, not on the probe being evaluated.
        val_cfg = ProbeTrainConfig(
            epochs=cfg["probes"]["epochs"],
            lr=cfg["probes"]["lr"],
            weight_decay=cfg["probes"]["weight_decay"],
            batch_size=cfg["probes"]["batch_size"],
        )
        val_probes = train_validation_probes(
            X_probe, zc_probe, ze_probe, val_cfg, device, min_acc=0.0,
        )
        print(f"[layer {layer}] val probes: zc={val_probes.acc_zc:.3f} "
              f"ze={val_probes.acc_ze:.3f}")

        interventions_dict = run_all_interventions(
            X_inter, zc_inter, val_probes.zc_probe, device, inter_cfg,
        )

        # K seeds × 3 architectures
        for arch in archs:
            for k in range(args.k):
                seed = 1000 + k  # deterministic seed schedule
                t1 = time.time()
                row = run_one_cell(
                    arch=arch, seed=seed,
                    X_probe=X_probe, zc_probe=zc_probe, ze_probe=ze_probe,
                    X_inter=X_inter, zc_inter=zc_inter, ze_inter=ze_inter,
                    X_test=X_test, zc_test=zc_test,
                    val_probes=val_probes,
                    interventions_dict=interventions_dict,
                    cfg=cfg, device=device,
                )
                row.update({
                    "model": cfg["model"]["name"],
                    "task": task.name,
                    "layer": layer,
                    "wallclock_s": time.time() - t1,
                })
                f_out.write(json.dumps(row) + "\n")
                f_out.flush()
                print(f"  L{layer} {arch:6s} seed={seed} "
                      f"acc={row['accuracy']:.3f} A={row['A']:.3f} "
                      f"R={row['R']:.3f} ({row['R_method']}) "
                      f"[{row['wallclock_s']:.1f}s]")

    f_out.close()
    print(f"\n✅ Benchmark complete: {out_path}")


if __name__ == "__main__":
    main()
