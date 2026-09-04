"""Run the post-registration robustness benchmark (one model x one task).

This runner addresses the construct-validity and dependence concerns that the
schema-v2 preregistered result was not designed to answer.  It does not modify
or append to ``results/benchmark_v2``.  Its five disjoint folds are:

    candidate_train  .30   train the candidate probe and its Hessian
    evaluator_train  .20   train bagged frozen target/control evaluators
    direction        .15   estimate edit directions; cross-fit diagnostics
    decoder_train    .20   train fresh decoders after each edit
    test             .15   final fixed- and retrained-decoder evaluation

For every candidate the runner reports raw and basis-invariant Hessian
alignment, architecture-size random-null normalization, eigenpair residuals,
orientation-robust fixed-decoder metrics, post-edit linear/MLP redecodability,
rank- and distortion-matched random controls, higher-rank Jacobian edits, and
direction stability.  Layer-level LEACE/INLP/RLACE/supervised-direction
baselines and FGSM/PGD/AlterRep magnitude sweeps are computed once per layer.

Usage:
    python -m scripts.run_robustness_v3 \
        --config configs/pythia.yaml --task sva --k 20 --deep-k 5

Smoke test:
    python -m scripts.run_robustness_v3 \
        --config configs/tiny.yaml --task sva --k 1 --deep-k 1 \
        --layers 6 --decoder-seeds 1 --direction-resamples 2 \
        --residual-top-n 1 --ranks 1
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from scripts.run_benchmark import ARCHS, default_data_paths, git_commit_hash, make_probe
from src.extraction import extract_all_layers, load_model, pick_device, select_layers
from src.gates import learnability_gate
from src.hessian import (
    compute_hessian_spectrum,
    hessian_eigen_residuals,
    random_subspace_null,
    subspace_alignment_sensitivity,
)
from src.interventions import (
    apply_alterrep,
    apply_fgsm,
    apply_pgd,
    inlp_projection,
    rlace_projection,
)
from src.metrics import train_validation_probes
from src.pipeline import load_config
from src.probes import ProbeTrainConfig, probe_accuracy, train_probe
from src.repro import hash_examples, set_seed
from src.robustness import (
    MatrixEdit,
    SubspaceEdit,
    cross_fitted_candidate_edit,
    direction_stability,
    distortion_matched_random_edit,
    distortion_norm,
    evaluate_fixed_evaluators,
    evaluate_retrained_decoders,
    example_group_id,
    fit_leace_binary,
    grouped_stratified_split,
    linear_probe_subspace,
    mean_evaluator_probabilities,
    orientation_robust_cs,
    position_variant,
    random_subspace,
    score_probe_binary,
    subspace_similarity,
)
from src.tasks import get_task
from src.ws5_repaired import (
    EVAL_MIN_ACC,
    EVAL_SEED_BASE,
    EvaluatorQualityError,
    candidate_direction_details,
    train_independent_evaluators,
)


SCHEMA_VERSION = 3
SPLIT_FRACS = {
    "candidate": 0.30,
    "evaluator": 0.20,
    "direction": 0.15,
    "decoder": 0.20,
    "test": 0.15,
}
DECODER_SEED_BASE = 710_000


def _csv_ints(value: str) -> list[int]:
    out = sorted(set(int(x.strip()) for x in value.split(",") if x.strip()))
    if not out or min(out) <= 0:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return out


def _csv_floats(value: str) -> list[float]:
    out = sorted(set(float(x.strip()) for x in value.split(",") if x.strip()))
    if not out or min(out) < 0:
        raise argparse.ArgumentTypeError("expected comma-separated non-negative numbers")
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--task", required=True, choices=["sva", "gender", "sst2"])
    p.add_argument("--k", type=int, default=20, help="candidate seeds per architecture/layer")
    p.add_argument("--deep-k", type=int, default=5,
                   help="first K candidate seeds per architecture receive expensive post-edit retraining")
    p.add_argument("--evaluators", type=int, default=5)
    p.add_argument("--decoder-seeds", type=int, default=1,
                   help="fresh-decoder initializations per deep candidate/edit")
    p.add_argument("--ranks", type=_csv_ints, default=_csv_ints("1,2,5"))
    p.add_argument("--hessian-ks", type=_csv_ints, default=_csv_ints("1,5,10,20"))
    p.add_argument("--hessian-null-draws", type=int, default=10_000)
    p.add_argument("--residual-top-n", type=int, default=5)
    p.add_argument("--residual-batch-size", type=int, default=256)
    p.add_argument("--direction-resamples", type=int, default=5)
    p.add_argument("--crossfit-folds", type=int, default=5)
    p.add_argument("--auc-floor", type=float, default=0.55)
    p.add_argument("--group-mode", default="auto",
                   choices=["auto", "none", "sentence", "minimal_pair", "occupation", "template"])
    p.add_argument("--position", default="last", choices=["last", "pre-pronoun", "post-pronoun"])
    p.add_argument("--perturbation-grid", type=_csv_floats,
                   default=_csv_floats("0.05,0.1,0.25,0.5,1.0"))
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--layers", default=None)
    p.add_argument("--data-paths", default=None)
    p.add_argument("--out-dir", default="results/robustness_v3")
    p.add_argument("--cache-dir", default="cache/robustness_v3")
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--skip-rlace", action="store_true",
                   help="skip only for development; final paper runs should include RLACE")
    return p.parse_args()


def _safe_position(position: str) -> str:
    return position.replace("-", "_")


def _load_jsonl_index(path: Path, key_fields: tuple[str, ...]) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != SCHEMA_VERSION:
                raise SystemExit(
                    f"{path}:{lineno}: expected schema {SCHEMA_VERSION}, got "
                    f"{row.get('schema_version')}; use a separate output directory"
                )
            key = tuple(row[field] for field in key_fields)
            if key in out:
                raise SystemExit(f"{path}:{lineno}: duplicate resume key {key}")
            out[key] = row
    return out


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()


def _decoder_seeds(n: int, offset: int = 0) -> list[int]:
    return [DECODER_SEED_BASE + offset + i for i in range(n)]


def _summarize_retrained(rows: list[dict]) -> dict:
    out = {}
    for feature in ("target", "control"):
        subset = [r for r in rows if r["feature"] == feature]
        aucs = [r["test"]["oriented_roc_auc"] for r in subset
                if math.isfinite(r["test"]["oriented_roc_auc"])]
        accs = [r["test"]["complemented_balanced_accuracy"] for r in subset
                if math.isfinite(r["test"]["complemented_balanced_accuracy"])]
        out[feature] = {
            "max_oriented_roc_auc": max(aucs) if aucs else None,
            "mean_oriented_roc_auc": sum(aucs) / len(aucs) if aucs else None,
            "max_complemented_balanced_accuracy": max(accs) if accs else None,
            "n_decoders": len(subset),
        }
    return out


def _evaluate_edit(
    edit,
    *,
    X_fit: torch.Tensor,
    X_decoder: torch.Tensor,
    zc_decoder: torch.Tensor,
    ze_decoder: torch.Tensor,
    X_test: torch.Tensor,
    zc_test: torch.Tensor,
    ze_test: torch.Tensor,
    evaluators,
    train_cfg: ProbeTrainConfig,
    device: torch.device,
    hidden_dim: int,
    decoder_seeds: list[int],
    auc_floor: float,
    retrain: bool,
) -> dict:
    X_decoder_post = edit.apply(X_decoder)
    X_test_post = edit.apply(X_test)
    fixed = evaluate_fixed_evaluators(
        evaluators, X_test, X_test_post, zc_test, ze_test, device,
        clean_auc_floor=auc_floor,
    )
    retrained = []
    if retrain:
        retrained = evaluate_retrained_decoders(
            X_decoder_post, zc_decoder, ze_decoder,
            X_test_post, zc_test, ze_test,
            train_cfg, device, hidden_dim=hidden_dim, seeds=decoder_seeds,
        )
    return {
        "edit": edit.metadata(),
        "distortion_fit_fro": distortion_norm(edit, X_fit),
        "distortion_test_fro": distortion_norm(edit, X_test),
        "fixed": fixed,
        "retrained": retrained,
        "retrained_summary": _summarize_retrained(retrained) if retrained else None,
    }


def _matched_pair_metrics(vp, X_pre, X_post, zc, ze, device, auc_floor):
    zc_pre = score_probe_binary(vp.zc_probe, X_pre, zc, device)
    zc_post = score_probe_binary(vp.zc_probe, X_post, zc, device)
    ze_pre = score_probe_binary(vp.ze_probe, X_pre, ze, device)
    ze_post = score_probe_binary(vp.ze_probe, X_post, ze, device)
    return {
        "target_pre": zc_pre.as_dict(),
        "target_post": zc_post.as_dict(),
        "control_pre": ze_pre.as_dict(),
        "control_post": ze_post.as_dict(),
        **orientation_robust_cs(zc_pre, zc_post, ze_pre, ze_post, clean_auc_floor=auc_floor),
    }


def _perturbation_sweep(
    attacker_pair,
    evaluators,
    X_test,
    zc_test,
    ze_test,
    device,
    grid,
    pgd_steps,
    auc_floor,
):
    rows = []
    for magnitude in grid:
        methods = {
            "AlterRep": apply_alterrep(
                X_test, zc_test, validation_probe=attacker_pair.zc_probe,
                device=device, alpha=magnitude,
            ),
            "FGSM": apply_fgsm(
                X_test, zc_test, validation_probe=attacker_pair.zc_probe,
                device=device, epsilon=magnitude,
            ),
            "PGD": apply_pgd(
                X_test, zc_test, validation_probe=attacker_pair.zc_probe,
                device=device, epsilon=magnitude, steps=pgd_steps,
                alpha=max(magnitude / max(pgd_steps // 2, 1), 1e-4),
            ),
        }
        for method, X_post in methods.items():
            rows.append({
                "method": method,
                "magnitude": float(magnitude),
                "matched": _matched_pair_metrics(
                    attacker_pair, X_test, X_post, zc_test, ze_test, device, auc_floor
                ),
                "split": evaluate_fixed_evaluators(
                    evaluators, X_test, X_post, zc_test, ze_test, device,
                    clean_auc_floor=auc_floor,
                ),
                "distortion_test_fro": float((X_post - X_test).norm()),
            })
    return rows


def _prediction_artifact(
    path: Path,
    test_examples,
    zc: torch.Tensor,
    ze: torch.Tensor,
    method_edits: dict,
    evaluators,
    X_test: torch.Tensor,
    device: torch.device,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        for method, edit in method_edits.items():
            X_post = edit.apply(X_test)
            probs = mean_evaluator_probabilities(evaluators, X_test, X_post, device)
            for i, ex in enumerate(test_examples):
                fh.write(json.dumps({
                    "method": method,
                    "example_id": hashlib.sha256(ex.sentence.encode("utf-8")).hexdigest()[:20],
                    "zc": int(zc[i]),
                    "ze": int(ze[i]),
                    "p_target_pre": float(probs["target_pre"][i]),
                    "p_target_post": float(probs["target_post"][i]),
                    "p_control_pre": float(probs["control_pre"][i]),
                    "p_control_post": float(probs["control_post"][i]),
                }, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _deduplicate(examples):
    seen = set()
    out = []
    for ex in examples:
        key = (ex.sentence, ex.zc, ex.ze)
        if key not in seen:
            seen.add(key)
            out.append(ex)
    return out


def main():
    args = parse_args()
    if args.deep_k > args.k:
        raise SystemExit("--deep-k cannot exceed --k")
    if args.decoder_seeds < 1:
        raise SystemExit("--decoder-seeds must be >= 1")

    config_path = Path(args.config) if Path(args.config).is_absolute() else PROJECT_ROOT / args.config
    cfg = load_config(config_path)
    if args.max_examples is not None:
        cfg["data"]["max_examples"] = args.max_examples
    if args.layers:
        cfg["extraction"]["layers"] = [int(x) for x in args.layers.split(",")]
    if max(args.hessian_ks) > cfg["hessian"]["top_n"]:
        cfg["hessian"]["top_n"] = max(args.hessian_ks)

    task = get_task(args.task)
    commit = git_commit_hash()
    set_seed(cfg["output"]["seed"])
    data_paths = (
        [Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p for p in args.data_paths.split(",")]
        if args.data_paths else default_data_paths(task.name, cfg)
    )
    examples = task.load(
        data_paths, max_examples=cfg["data"].get("max_examples"), seed=cfg["data"]["seed"]
    )
    examples = position_variant(_deduplicate(examples), task.name, args.position)
    folds, split_diagnostics = grouped_stratified_split(
        examples, SPLIT_FRACS, seed=cfg["data"]["seed"],
        task_name=task.name, group_mode=args.group_mode,
    )

    print("=" * 78)
    print(f"  Robustness v{SCHEMA_VERSION}: {cfg['model']['name']} x {task.name} "
          f"position={args.position} k={args.k} deep_k={args.deep_k}")
    print("=" * 78)
    print(f"[split] {split_diagnostics['fold_sizes']} groups={split_diagnostics['n_groups']} "
          f"mode={args.group_mode}")

    device = pick_device()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[cfg["model"]["dtype"]]
    bundle = load_model(
        cfg["model"]["name"], device=device, dtype=dtype,
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    )
    explicit = cfg["extraction"].get("layers")
    layers = list(explicit) if explicit else select_layers(
        bundle.n_layers, k=cfg["extraction"]["num_layers_to_probe"]
    )
    print(f"[layers] {layers}")

    safe_model = cfg["model"]["name"].replace("/", "_")
    prefix = f"{safe_model}_{task.name}_{_safe_position(args.position)}"
    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = out_dir / f"{prefix}.candidates.jsonl"
    layer_path = out_dir / f"{prefix}.layers.jsonl"
    manifest_path = out_dir / f"{prefix}.manifest.json"
    predictions_dir = out_dir / "predictions" / prefix
    done_candidates = _load_jsonl_index(candidate_path, ("layer", "arch", "seed"))
    done_layers = _load_jsonl_index(layer_path, ("layer",))

    def candidate_layer_done(layer):
        if (layer,) in done_layers and done_layers[(layer,)].get("status") == "excluded":
            return True
        return all((layer, arch, 1000 + i) in done_candidates
                   for arch in ARCHS for i in range(args.k))

    pending_layers = [layer for layer in layers if not candidate_layer_done(layer)]
    if not pending_layers:
        print(f"[done] all requested cells already present in {candidate_path}")
        return
    gate_layer = layers[len(layers) // 2]
    extraction_layers = sorted(set(pending_layers) | {gate_layer})

    cache_root = None
    if args.cache_dir:
        cache_root = (
            Path(args.cache_dir) if Path(args.cache_dir).is_absolute()
            else PROJECT_ROOT / args.cache_dir
        ) / prefix
    reps = {}
    for fold_name, fold_examples in folds.items():
        print(f"[extract] {fold_name}: n={len(fold_examples)} layers={extraction_layers}")
        reps[fold_name] = extract_all_layers(
            bundle, fold_examples, extraction_layers,
            batch_size=cfg["extraction"]["batch_size"],
            max_length=cfg["extraction"]["max_length"],
            cache_dir=cache_root, cache_tag=fold_name,
        )

    # Candidate-fold z-scoring, reused exactly across all five folds.
    standardization = {}
    for layer in extraction_layers:
        X_candidate = reps["candidate"][layer]["X"]
        mean = X_candidate.mean(dim=0, keepdim=True)
        sd = X_candidate.std(dim=0, keepdim=True).clamp_min(1e-6)
        standardization[layer] = {
            "mean_abs": float(mean.abs().mean()),
            "sd_min": float(sd.min()),
            "sd_median": float(sd.median()),
        }
        for fold_name in folds:
            reps[fold_name][layer]["X"] = (reps[fold_name][layer]["X"] - mean) / sd

    train_cfg = ProbeTrainConfig(
        epochs=cfg["probes"]["epochs"], lr=cfg["probes"]["lr"],
        weight_decay=cfg["probes"]["weight_decay"],
        batch_size=cfg["probes"]["batch_size"],
    )
    gate_rep = reps["candidate"][gate_layer]
    gate = learnability_gate(
        gate_rep["X"], gate_rep["zc"], gate_rep["ze"], device, train_cfg,
        zc_floor=task.zc_gate_floor, ze_floor=task.ze_gate_floor,
        chance=task.chance_accuracy, layer=gate_layer,
    )
    print(f"[gate] {gate.report()}")

    manifest = {"schema_version": SCHEMA_VERSION, "runs": []}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            raise SystemExit(f"unreadable manifest: {manifest_path}")
    manifest["runs"].append({
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "config_file": str(args.config),
        "config": cfg,
        "task": task.name,
        "position": args.position,
        "group_mode": args.group_mode,
        "split_fracs": SPLIT_FRACS,
        "split_diagnostics": split_diagnostics,
        "data_paths": [str(p) for p in data_paths],
        "data_hash": hash_examples(examples),
        "n_examples": len(examples),
        "layers": layers,
        "k": args.k,
        "deep_k": args.deep_k,
        "deep_subset_rule": "seed_index < deep_k within each layer x architecture",
        "evaluators": args.evaluators,
        "decoder_seeds": args.decoder_seeds,
        "ranks": args.ranks,
        "hessian_ks": args.hessian_ks,
        "eigenpair_residual_rule": "seed_index == 0 within each layer x architecture",
        "auc_floor": args.auc_floor,
        "gate": gate.as_dict(),
        "post_registered": True,
        "resumed_candidates": len(done_candidates),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    if not gate.passed:
        for layer in layers:
            if (layer,) not in done_layers:
                _append_jsonl(layer_path, {
                    "schema_version": SCHEMA_VERSION,
                    "git_commit": commit,
                    "model": cfg["model"]["name"],
                    "task": task.name,
                    "position": args.position,
                    "layer": layer,
                    "status": "excluded",
                    "reason": "learnability_gate_failed",
                    "gate": gate.as_dict(),
                })
        print("[excluded] model-task failed the learnability gate; no candidate rows written")
        raise SystemExit(2)

    decoder_seeds = _decoder_seeds(args.decoder_seeds)
    n_written = 0
    for layer in pending_layers:
        layer_start = time.time()
        fold_rep = {name: reps[name][layer] for name in folds}
        Xc, zcc = fold_rep["candidate"]["X"], fold_rep["candidate"]["zc"]
        Xe = fold_rep["evaluator"]["X"]
        zce, zee = fold_rep["evaluator"]["zc"], fold_rep["evaluator"]["ze"]
        Xd = fold_rep["direction"]["X"]
        zcd, zed = fold_rep["direction"]["zc"], fold_rep["direction"]["ze"]
        Xr = fold_rep["decoder"]["X"]
        zcr, zer = fold_rep["decoder"]["zc"], fold_rep["decoder"]["ze"]
        Xt = fold_rep["test"]["X"]
        zct, zet = fold_rep["test"]["zc"], fold_rep["test"]["ze"]

        try:
            evaluators = train_independent_evaluators(
                Xe, zce, zee, train_cfg, device,
                n_evaluators=args.evaluators, min_acc=EVAL_MIN_ACC,
                seed_base=EVAL_SEED_BASE + 101 * layer,
            )
        except EvaluatorQualityError as exc:
            if (layer,) not in done_layers:
                # Preserve a candidate-side selection diagnostic even though
                # the reliability target is undefined. This lets the analysis
                # compare excluded and included layers instead of conditioning
                # on evaluator success without any observable check.
                selection_diagnostic = None
                selection_error = None
                try:
                    set_seed(1000)
                    selection_probe = make_probe(
                        "linear", Xc.shape[1],
                        hidden_dim=cfg["probes"]["hidden_dim"],
                        mka_lambda=cfg["mka"]["lambda_reg"],
                        knn_k=cfg["mka"]["knn_k"], seed=1000,
                    ).to(device)
                    train_probe(selection_probe, Xc, zcc, train_cfg, device)
                    selection_spectrum = compute_hessian_spectrum(
                        selection_probe, Xc, zcc, device,
                        top_n=cfg["hessian"]["top_n"], bottom_n=0,
                        max_iter=cfg["hessian"]["max_iter"],
                        tol=cfg["hessian"]["tol"],
                    )
                    selection_diagnostic = {
                        "arch": "linear", "seed": 1000,
                        "candidate_accuracy": probe_accuracy(
                            selection_probe, Xt, zct, device
                        ),
                        "A_raw_mean_abs_cosine": selection_spectrum.mean_align_top,
                        "A_subspace": subspace_alignment_sensitivity(
                            selection_probe, selection_spectrum.eigvecs_top,
                            ks=args.hessian_ks,
                        ),
                    }
                except Exception as diagnostic_exc:
                    selection_error = repr(diagnostic_exc)
                _append_jsonl(layer_path, {
                    "schema_version": SCHEMA_VERSION,
                    "git_commit": commit,
                    "model": cfg["model"]["name"],
                    "task": task.name,
                    "position": args.position,
                    "layer": layer,
                    "status": "excluded",
                    "reason": "evaluator_quality_failed",
                    "detail": str(exc),
                    "selection_diagnostic": selection_diagnostic,
                    "selection_diagnostic_error": selection_error,
                })
            print(f"[layer {layer}] excluded: {exc}")
            continue

        # Layer-level methods are candidate-independent and are evaluated once.
        if (layer,) not in done_layers:
            set_seed(800_000 + layer)
            attacker_pair = train_validation_probes(
                Xd, zcd, zed, train_cfg, device, min_acc=0.0
            )
            supervised_edit = SubspaceEdit(linear_probe_subspace(attacker_pair.zc_probe))
            random_rank1 = SubspaceEdit(random_subspace(Xd.shape[1], 1, 810_000 + layer))
            random_matched, random_matched_meta = distortion_matched_random_edit(
                supervised_edit, Xd, seed=820_000 + layer
            )
            layer_edits = {
                "supervised_linear_direction": supervised_edit,
                "LEACE": fit_leace_binary(Xd, zcd),
                "INLP": MatrixEdit(inlp_projection(
                    Xd, zcd, device=device,
                    num_iters=cfg["interventions"]["inlp"]["num_iters"],
                )),
                "random_rank1": random_rank1,
                "random_distortion_matched": random_matched,
            }
            if max(args.ranks) > 1:
                layer_edits[f"random_rank{max(args.ranks)}"] = SubspaceEdit(
                    random_subspace(Xd.shape[1], min(max(args.ranks), Xd.shape[1]), 830_000 + layer)
                )
            if not args.skip_rlace:
                layer_edits["RLACE_approx"] = MatrixEdit(rlace_projection(
                    Xd, zcd, device=device,
                    rank=cfg["interventions"]["rlace"]["rank"],
                    steps=cfg["interventions"]["rlace"]["steps"],
                ))
            layer_baselines = {
                name: _evaluate_edit(
                    edit, X_fit=Xd, X_decoder=Xr, zc_decoder=zcr, ze_decoder=zer,
                    X_test=Xt, zc_test=zct, ze_test=zet, evaluators=evaluators,
                    train_cfg=train_cfg, device=device,
                    hidden_dim=cfg["probes"]["hidden_dim"],
                    decoder_seeds=decoder_seeds, auc_floor=args.auc_floor,
                    retrain=True,
                )
                for name, edit in layer_edits.items()
            }
            clean_retrained = evaluate_retrained_decoders(
                Xr, zcr, zer, Xt, zct, zet, train_cfg, device,
                hidden_dim=cfg["probes"]["hidden_dim"], seeds=decoder_seeds,
            )
            sweep = _perturbation_sweep(
                attacker_pair, evaluators, Xt, zct, zet, device,
                args.perturbation_grid,
                pgd_steps=cfg["interventions"]["pgd"]["steps"],
                auc_floor=args.auc_floor,
            )
            _append_jsonl(layer_path, {
                "schema_version": SCHEMA_VERSION,
                "git_commit": commit,
                "model": cfg["model"]["name"],
                "task": task.name,
                "position": args.position,
                "layer": layer,
                "status": "included",
                "fold_sizes": split_diagnostics["fold_sizes"],
                "standardization": standardization[layer],
                "evaluator_acc_zc_oos": [ev.acc_zc_oos for ev in evaluators],
                "evaluator_acc_ze_oos": [ev.acc_ze_oos for ev in evaluators],
                "clean_retrained": clean_retrained,
                "clean_retrained_summary": _summarize_retrained(clean_retrained),
                "baselines": layer_baselines,
                "random_distortion_match": random_matched_meta,
                "perturbation_sweep": sweep,
                "wallclock_s": time.time() - layer_start,
            })

        for arch in ARCHS:
            candidate_ranks = [1] if arch == "linear" else args.ranks
            for seed_index in range(args.k):
                seed = 1000 + seed_index
                key = (layer, arch, seed)
                if key in done_candidates:
                    continue
                started = time.time()
                set_seed(seed)
                probe = make_probe(
                    arch, Xc.shape[1], hidden_dim=cfg["probes"]["hidden_dim"],
                    mka_lambda=cfg["mka"]["lambda_reg"],
                    knn_k=cfg["mka"]["knn_k"], seed=seed,
                ).to(device)
                train_probe(probe, Xc, zcc, train_cfg, device)
                candidate_test_scores = score_probe_binary(probe, Xt, zct, device).as_dict()

                spectrum = compute_hessian_spectrum(
                    probe, Xc, zcc, device,
                    top_n=cfg["hessian"]["top_n"], bottom_n=0,
                    max_iter=cfg["hessian"]["max_iter"], tol=cfg["hessian"]["tol"],
                )
                subspace = subspace_alignment_sensitivity(
                    probe, spectrum.eigvecs_top, ks=args.hessian_ks
                )
                nulls = {}
                for k_str, observed in subspace.items():
                    effective_k = min(int(k_str), len(spectrum.eigvecs_top), probe.num_params())
                    nulls[k_str] = random_subspace_null(
                        observed, probe.num_params(), effective_k,
                        draws=args.hessian_null_draws,
                        seed=seed + 10_000 * layer + effective_k,
                    )
                residuals = hessian_eigen_residuals(
                    probe, Xc, zcc, spectrum.eigenvalues_top, spectrum.eigvecs_top,
                    device, top_n=args.residual_top_n,
                    batch_size=args.residual_batch_size,
                ) if args.residual_top_n > 0 and seed_index == 0 else []

                deep = seed_index < args.deep_k
                candidate_edits = {}
                edit_objects = {}
                max_rank_detail = candidate_direction_details(
                    probe, Xd, device, r=max(candidate_ranks)
                )
                for rank in candidate_ranks:
                    if max_rank_detail["Q"] is None:
                        detail = {
                            **{k: v for k, v in max_rank_detail.items() if k != "Q"},
                            "Q": None,
                            "requested_rank": rank,
                            "effective_rank": 0,
                            "singular_values": max_rank_detail["singular_values"][:rank],
                        }
                    else:
                        effective_rank = min(
                            rank,
                            max_rank_detail["effective_rank"],
                            max_rank_detail["Q"].shape[1],
                        )
                        detail = {
                            **{k: v for k, v in max_rank_detail.items() if k != "Q"},
                            "Q": max_rank_detail["Q"][:, :effective_rank],
                            "requested_rank": rank,
                            "effective_rank": effective_rank,
                            "singular_values": max_rank_detail["singular_values"][:rank],
                        }
                    name = f"candidate_r{rank}"
                    record = {k: v for k, v in detail.items() if k != "Q"}
                    if detail["Q"] is None:
                        record.update({"degenerate": True, "evaluation": None})
                        candidate_edits[name] = record
                        continue
                    edit = SubspaceEdit(detail["Q"])
                    edit_objects[name] = edit
                    record.update({
                        "degenerate": False,
                        "evaluation": _evaluate_edit(
                            edit, X_fit=Xd, X_decoder=Xr, zc_decoder=zcr, ze_decoder=zer,
                            X_test=Xt, zc_test=zct, ze_test=zet, evaluators=evaluators,
                            train_cfg=train_cfg, device=device,
                            hidden_dim=cfg["probes"]["hidden_dim"],
                            decoder_seeds=decoder_seeds, auc_floor=args.auc_floor,
                            retrain=deep,
                        ),
                    })
                    candidate_edits[name] = record

                    random_edit = SubspaceEdit(random_subspace(Xd.shape[1], edit.rank, seed + 90_000 + rank))
                    random_name = f"random_rank_matched_r{rank}"
                    edit_objects[random_name] = random_edit
                    candidate_edits[random_name] = {
                        "evaluation": _evaluate_edit(
                            random_edit, X_fit=Xd, X_decoder=Xr, zc_decoder=zcr, ze_decoder=zer,
                            X_test=Xt, zc_test=zct, ze_test=zet, evaluators=evaluators,
                            train_cfg=train_cfg, device=device,
                            hidden_dim=cfg["probes"]["hidden_dim"],
                            decoder_seeds=decoder_seeds, auc_floor=args.auc_floor,
                            retrain=False,
                        )
                    }
                    matched_edit, matched_meta = distortion_matched_random_edit(
                        edit, Xd, seed=seed + 190_000 + rank
                    )
                    matched_name = f"random_distortion_matched_r{rank}"
                    edit_objects[matched_name] = matched_edit
                    candidate_edits[matched_name] = {
                        "match": matched_meta,
                        "evaluation": _evaluate_edit(
                            matched_edit, X_fit=Xd, X_decoder=Xr, zc_decoder=zcr, ze_decoder=zer,
                            X_test=Xt, zc_test=zct, ze_test=zet, evaluators=evaluators,
                            train_cfg=train_cfg, device=device,
                            hidden_dim=cfg["probes"]["hidden_dim"],
                            decoder_seeds=decoder_seeds, auc_floor=args.auc_floor,
                            retrain=False,
                        ),
                    }

                if deep:
                    stability = direction_stability(
                        probe, Xd, device, rank=1,
                        resamples=args.direction_resamples,
                        seed=seed + 300_000 + layer,
                    )
                    crossfit = cross_fitted_candidate_edit(
                        probe, Xd, device, rank=1,
                        folds=min(args.crossfit_folds, max(2, Xd.shape[0] // 2)),
                        seed=seed + 400_000 + layer,
                    )
                    crossfit_eval = None
                    if crossfit["X_post"] is not None:
                        crossfit_eval = evaluate_fixed_evaluators(
                            evaluators, Xd, crossfit["X_post"], zcd, zed, device,
                            clean_auc_floor=args.auc_floor,
                        )
                    crossfit_record = {
                        k: v for k, v in crossfit.items() if k != "X_post"
                    }
                    crossfit_record["fixed"] = crossfit_eval
                    crossfit_record["evaluated"] = True
                else:
                    stability = {"evaluated": False}
                    crossfit_record = {"evaluated": False}

                candidate_q = edit_objects.get("candidate_r1")
                evaluator_agreements = []
                if candidate_q is not None:
                    evaluator_agreements = [
                        subspace_similarity(
                            candidate_q.Q,
                            linear_probe_subspace(ev.probes.zc_probe),
                        )
                        for ev in evaluators
                    ]

                prediction_file = None
                if args.save_predictions and deep and edit_objects:
                    prediction_path = predictions_dir / f"L{layer}_{arch}_{seed}.jsonl.gz"
                    _prediction_artifact(
                        prediction_path, folds["test"], zct, zet,
                        {name: edit for name, edit in edit_objects.items()
                         if name.startswith("candidate_")},
                        evaluators, Xt, device,
                    )
                    prediction_file = str(prediction_path.relative_to(PROJECT_ROOT))

                row = {
                    "schema_version": SCHEMA_VERSION,
                    "git_commit": commit,
                    "post_registered": True,
                    "model": cfg["model"]["name"],
                    "task": task.name,
                    "position": args.position,
                    "group_mode": args.group_mode,
                    "layer": layer,
                    "arch": arch,
                    "seed": seed,
                    "deep_evaluated": deep,
                    "candidate_accuracy": probe_accuracy(probe, Xt, zct, device),
                    "candidate_test_scores": candidate_test_scores,
                    "hessian": {
                        "A_raw_mean_abs_cosine": spectrum.mean_align_top,
                        "lambda_max": spectrum.lambda_max,
                        "eigenvalues_top": [float(x) for x in spectrum.eigenvalues_top],
                        "align_top": [float(x) for x in spectrum.align_top],
                        "A_subspace": subspace,
                        "random_null": nulls,
                        "eigenpair_residuals": residuals,
                    },
                    "direction_stability": stability,
                    "candidate_evaluator_agreement": {
                        "values": evaluator_agreements,
                        "mean": (sum(evaluator_agreements) / len(evaluator_agreements)
                                 if evaluator_agreements else None),
                        "median": (statistics.median(evaluator_agreements)
                                   if evaluator_agreements else None),
                    },
                    "crossfit": crossfit_record,
                    "edits": candidate_edits,
                    "prediction_artifact": prediction_file,
                    "num_params": probe.num_params(),
                    "fold_sizes": split_diagnostics["fold_sizes"],
                    "wallclock_s": time.time() - started,
                }
                _append_jsonl(candidate_path, row)
                n_written += 1
                fixed_r = candidate_edits.get("candidate_r1", {}).get("evaluation", {})
                fixed_r = fixed_r.get("fixed", {}).get("aggregate", {}).get("R_auc_oriented")
                r_text = "null" if fixed_r is None else f"{fixed_r:.3f}"
                print(f"  L{layer} {arch:6s} s={seed} deep={deep} "
                      f"acc={row['candidate_accuracy']:.3f} "
                      f"Araw={spectrum.mean_align_top:.3f} Rauc={r_text} "
                      f"[{row['wallclock_s']:.1f}s]")

    print(f"[done] wrote {n_written} candidate rows -> {candidate_path}")
    print(f"[layers] inclusion/baselines -> {layer_path}")
    print(f"[manifest] {manifest_path}")


if __name__ == "__main__":
    main()
