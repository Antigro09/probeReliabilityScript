"""
K-seed benchmark runner for the alignment-as-predictor experiment (v2).

For each (model, layer, task, architecture) combination, train K probes
with different random seeds. For each probe, record:
    - test accuracy
    - mean directional alignment with top-20 Hessian eigenvectors (the predictor A)
    - reliability across the five intervention methods (the target R),
      with the FULL per-method metric record (raw pre/post validation
      accuracies, unclipped ratios, room denominators, floored + legacy
      clipped values — see src/metrics.py)

Usage:
    # Run the full benchmark (slow, days of compute):
    python -m scripts.run_benchmark --config configs/pythia.yaml --task sva --k 20

    # Quick smoke test (single layer, k=2):
    python -m scripts.run_benchmark --config configs/tiny.yaml --task sva --k 2

Output:
    <out-dir>/<model>_<task>.jsonl        (default out-dir: results/benchmark_v2)
    one line per probe = one (model, layer, task, architecture, seed) cell,
    stamped with schema_version and the git commit of the code that wrote it.
    <out-dir>/<model>_<task>.manifest.json  records config, data hashes,
    learnability-gate results, and one entry per (re)launch.

v2 hardening (WS0):
    - RESUME-SAFE: on startup the existing output file is read and completed
      (layer, arch, seed) cells are skipped. v1 opened the file in append
      mode with no check, so an interrupted-and-restarted run silently
      duplicated rows and poisoned every median.
    - VERSIONED OUTPUT: v2 rows land in their own directory and carry
      schema_version + git commit; analysis scripts take an explicit input
      dir instead of globbing everything (v1 blended any *.jsonl it found).
    - LEARNABILITY GATE (WS2): before launching, a linear probe at the
      middle probed layer must beat the task's registered floor and a
      shuffled-label control must sit at chance, for both Zc and Ze.
      Otherwise the run refuses to start (exit code 2).
    - SINGLE-PASS EXTRACTION + CACHE: all probed layers are captured from
      one forward pass per batch and cached to disk with provenance
      (v1 ran a full forward per layer and cached nothing here).

The JSONL format keeps individual seeds visible so the predictor evaluator
can compute median-over-seeds aggregations honestly.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.pipeline import load_config
from src.extraction import (
    load_model, select_layers, extract_all_layers, pick_device,
)
from src.probes import (
    LinearProbe, MLPProbe, MKAProbe,
    ProbeTrainConfig, train_probe, probe_accuracy,
)
from src.metrics import (
    train_validation_probes, compute_intervention_metrics, DECODABILITY_FLOOR,
)
from src.gates import learnability_gate
from src.interventions import InterventionConfig, run_all_interventions
from src.hessian import compute_hessian_spectrum
from src.repro import set_seed, hash_examples
from src.tasks import get_task

SCHEMA_VERSION = 2
ARCHS = ("linear", "mlp", "mka")


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
    p.add_argument("--out-dir", default="results/benchmark_v2",
                   help="Output directory (versioned; keep v2 rows away "
                        "from the tainted v1 records)")
    p.add_argument("--cache-dir", default="cache/benchmark_v2",
                   help="Representation cache directory ('' disables caching)")
    p.add_argument("--floor", type=float, default=DECODABILITY_FLOOR,
                   help="Decodability floor for C/S denominators "
                        "(see src/metrics.py; lock in PREREGISTRATION_v2.md)")
    return p.parse_args()


def git_commit_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
    except Exception:
        return "unknown"


def default_data_paths(task_name: str, cfg: dict) -> list[Path]:
    """Return task-specific default data paths.

    Model configs point at the original SVA files. Other tasks have their
    own conventional filenames; missing files are a hard error inside the
    task loaders (there are deliberately no silent fallbacks).
    """
    if task_name == "gender":
        return [PROJECT_ROOT / "data" / "gender.tsv"]
    if task_name == "sst2":
        return [PROJECT_ROOT / "data" / "sst2.tsv"]
    return [PROJECT_ROOT / p for p in cfg["data"]["paths"]]


def load_completed_cells(out_path: Path) -> set[tuple[int, str, int]]:
    """Read an existing output file and return completed (layer, arch, seed).

    Refuses to resume onto a file containing rows from a different schema
    version — mixed-schema files are how tainted and clean data blend
    invisibly.
    """
    done: set[tuple[int, str, int]] = set()
    if not out_path.exists():
        return done
    with out_path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row_schema = row.get("schema_version", 1)
            if row_schema != SCHEMA_VERSION:
                raise SystemExit(
                    f"{out_path}:{lineno}: row has schema_version="
                    f"{row_schema}, this runner writes {SCHEMA_VERSION}. "
                    f"Refusing to mix schema versions in one file — move the "
                    f"old file out of the way or pass a different --out-dir."
                )
            key = (row["layer"], row["arch"], row["seed"])
            if key in done:
                raise SystemExit(
                    f"{out_path}:{lineno}: duplicate cell {key} already in "
                    f"the output file. This file predates the resume fix or "
                    f"was concatenated; de-duplicate before resuming."
                )
            done.add(key)
    return done


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
    floor: float,
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

    # ---- Target R across the 5 interventions ----
    # Important: re-training validation probes WOULD leak. We use the
    # pre-computed val_probes (trained on un-intervened reps per-layer).
    # Schema v2 stores the FULL metric record per method (raw pre/post
    # accuracies, unclipped ratios, rooms) so denominator pathologies are
    # diagnosable from the record itself (see src/metrics.py).
    per_method: dict[str, dict] = {}
    best_R, best_method = None, None
    best_R_legacy, best_method_legacy = -1.0, None
    for method, X_post in interventions_dict.items():
        m = compute_intervention_metrics(
            val_probes, X_pre=X_inter, X_post=X_post,
            zc=zc_inter, ze=ze_inter, device=device, floor=floor,
        )
        per_method[method] = m.as_dict()
        if m.reliability is not None and \
                (best_R is None or m.reliability > best_R):
            best_R = m.reliability
            best_method = method
        if m.reliability_clipped > best_R_legacy:
            best_R_legacy = m.reliability_clipped
            best_method_legacy = method

    return {
        "arch": arch,
        "seed": seed,
        "accuracy": acc,
        "A": A,
        "lambda_max": lambda_max,
        # Floored (v2) target: None if EVERY method's R is null for this
        # cell (e.g. Ze not decodable at this layer).
        "R": best_R,
        "R_method": best_method,
        # Legacy v1 semantics (max over clipped R), for comparability only.
        "R_legacy": best_R_legacy,
        "R_legacy_method": best_method_legacy,
        "per_method": per_method,
        "val_zc_acc": val_probes.acc_zc,
        "val_ze_acc": val_probes.acc_ze,
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
    print("=" * 70)
    print(f"  Benchmark v{SCHEMA_VERSION}: {cfg['model']['name']}  "
          f"task={task.name}  k={args.k}  commit={commit[:12]}")
    print("=" * 70)

    set_seed(cfg["output"]["seed"])

    # Data paths -- use task-specific overrides if provided, else config defaults.
    if args.data_paths:
        data_paths = [Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p
                      for p in args.data_paths.split(",")]
    else:
        data_paths = default_data_paths(task.name, cfg)

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

    # Output (versioned, resume-safe)
    safe_name = cfg["model"]["name"].replace("/", "_")
    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() \
        else PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_name}_{task.name}.jsonl"
    manifest_path = out_dir / f"{safe_name}_{task.name}.manifest.json"
    done = load_completed_cells(out_path)
    total_cells = len(layers) * len(ARCHS) * args.k
    print(f"[output] {out_path}")
    if done:
        print(f"[resume] found {len(done)} completed cells "
              f"({total_cells - len(done)} remaining)")

    cache_dir = None
    if args.cache_dir:
        cache_dir = (Path(args.cache_dir) if Path(args.cache_dir).is_absolute()
                     else PROJECT_ROOT / args.cache_dir) / f"{safe_name}_{task.name}"

    def layer_done(layer: int) -> bool:
        return all((layer, a, 1000 + k) in done
                   for a in ARCHS for k in range(args.k))

    pending_layers = [l for l in layers if not layer_done(l)]
    gate_layer = layers[len(layers) // 2]

    # ---- Extraction: one forward pass per batch for all needed layers ----
    # The gate layer's probe-split reps are always needed, even on resume.
    probe_layers = sorted(set(pending_layers) | {gate_layer})
    print(f"[extract] probe-train split, layers {probe_layers}")
    reps_probe = extract_all_layers(
        bundle, train_ex, probe_layers,
        batch_size=cfg["extraction"]["batch_size"],
        max_length=cfg["extraction"]["max_length"],
        cache_dir=cache_dir, cache_tag="probe",
    )
    print(f"[extract] intervention split, layers {pending_layers}")
    reps_inter = extract_all_layers(
        bundle, inter_ex, pending_layers,
        batch_size=cfg["extraction"]["batch_size"],
        max_length=cfg["extraction"]["max_length"],
        cache_dir=cache_dir, cache_tag="inter",
    )
    print(f"[extract] test split, layers {pending_layers}")
    reps_test = extract_all_layers(
        bundle, test_ex, pending_layers,
        batch_size=cfg["extraction"]["batch_size"],
        max_length=cfg["extraction"]["max_length"],
        cache_dir=cache_dir, cache_tag="test",
    )

    train_cfg = ProbeTrainConfig(
        epochs=cfg["probes"]["epochs"],
        lr=cfg["probes"]["lr"],
        weight_decay=cfg["probes"]["weight_decay"],
        batch_size=cfg["probes"]["batch_size"],
    )

    # ---- Learnability gate (WS2): refuse to launch on an unlearnable task ----
    g = reps_probe[gate_layer]
    gate = learnability_gate(
        g["X"], g["zc"], g["ze"], device, train_cfg,
        zc_floor=task.zc_gate_floor, ze_floor=task.ze_gate_floor,
        chance=task.chance_accuracy, layer=gate_layer,
    )
    print(f"[gate] {gate.report()}")

    # ---- Manifest: provenance for this (re)launch ----
    manifest = {"schema_version": SCHEMA_VERSION, "runs": []}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            print(f"[warn] unreadable manifest {manifest_path}; rewriting")
            manifest = {"schema_version": SCHEMA_VERSION, "runs": []}
    manifest["runs"].append({
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
                         .isoformat(timespec="seconds"),
        "git_commit": commit,
        "config_file": str(args.config),
        "config": cfg,
        "task": task.name,
        "k": args.k,
        "layers": layers,
        "floor": args.floor,
        "gate": gate.as_dict(),
        "data_paths": [str(p) for p in data_paths],
        "data_hash_all": hash_examples(examples),
        "n_examples": len(examples),
        "resumed_completed_cells": len(done),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if not gate.passed:
        print("\n❌ LEARNABILITY GATE FAILED — refusing to launch the "
              "benchmark. A probe cannot decode the task's features from "
              "clean representations (or the shuffled-label control beat "
              "chance, indicating leakage). Fix the task data; see "
              "src/gates.py and the gate record in the manifest:")
        print(f"   {manifest_path}")
        sys.exit(2)

    if not pending_layers:
        print("\n✅ Nothing to do: all cells already completed "
              f"({len(done)}/{total_cells}). Output: {out_path}")
        return

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

    n_written = 0
    n_skipped = 0
    # Append is safe now: completed cells were read into `done` above.
    with out_path.open("a") as f_out:
        for layer in pending_layers:
            X_probe = reps_probe[layer]["X"]
            zc_probe = reps_probe[layer]["zc"]
            ze_probe = reps_probe[layer]["ze"]
            X_inter = reps_inter[layer]["X"]
            zc_inter = reps_inter[layer]["zc"]
            ze_inter = reps_inter[layer]["ze"]
            X_test = reps_test[layer]["X"]
            zc_test = reps_test[layer]["zc"]

            # Validation probes & interventions are computed ONCE per layer.
            # They depend only on the representations, not on the probe
            # being evaluated.
            set_seed(cfg["output"]["seed"])
            val_probes = train_validation_probes(
                X_probe, zc_probe, ze_probe, train_cfg, device, min_acc=0.0,
            )
            print(f"[layer {layer}] val probes: zc={val_probes.acc_zc:.3f} "
                  f"ze={val_probes.acc_ze:.3f}")

            interventions_dict = run_all_interventions(
                X_inter, zc_inter, val_probes.zc_probe, device, inter_cfg,
            )

            # K seeds × 3 architectures
            for arch in ARCHS:
                for k in range(args.k):
                    seed = 1000 + k  # deterministic seed schedule
                    if (layer, arch, seed) in done:
                        n_skipped += 1
                        continue
                    t1 = time.time()
                    row = run_one_cell(
                        arch=arch, seed=seed,
                        X_probe=X_probe, zc_probe=zc_probe, ze_probe=ze_probe,
                        X_inter=X_inter, zc_inter=zc_inter, ze_inter=ze_inter,
                        X_test=X_test, zc_test=zc_test,
                        val_probes=val_probes,
                        interventions_dict=interventions_dict,
                        cfg=cfg, device=device, floor=args.floor,
                    )
                    row.update({
                        "model": cfg["model"]["name"],
                        "task": task.name,
                        "layer": layer,
                        "wallclock_s": time.time() - t1,
                        "schema_version": SCHEMA_VERSION,
                        "git_commit": commit,
                    })
                    f_out.write(json.dumps(row) + "\n")
                    f_out.flush()
                    n_written += 1
                    r_str = "null" if row["R"] is None else f"{row['R']:.3f}"
                    print(f"  L{layer} {arch:6s} seed={seed} "
                          f"acc={row['accuracy']:.3f} A={row['A']:.3f} "
                          f"R={r_str} ({row['R_method']}) "
                          f"[{row['wallclock_s']:.1f}s]")

    print(f"\n✅ Benchmark complete: {out_path}")
    print(f"   wrote {n_written} new cells, skipped {n_skipped} completed, "
          f"total now {len(done) + n_written}/{total_cells}")


if __name__ == "__main__":
    main()
