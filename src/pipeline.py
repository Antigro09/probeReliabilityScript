"""
End-to-end pipeline: from a YAML config + Linzen data to a results dict
that contains everything for one (model, layer) combination.

Workflow per layer:
    1. Extract representations
    2. Train Linear / MLP / MKA probes
    3. Train Zc and Ze validation probes
    4. Compute probe accuracies
    5. Apply all five interventions
    6. Compute completeness / selectivity / reliability per intervention
    7. Compute Hessian spectrum + directional alignment for each probe
    8. Compute MKA score between original X and probe-induced hidden states

The output is a dict of dicts that's fully JSON-serializable, ready for
aggregation across models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from .data import build_examples, split_balanced
from .extraction import (
    load_model, select_layers, extract_layer_reps, pick_device,
)
from .probes import (
    LinearProbe, MLPProbe, MKAProbe,
    ProbeTrainConfig, train_probe, probe_accuracy,
    knn_kernel,
)
from .probes import mka_score as mka_score_fn
from .metrics import (
    train_validation_probes, compute_intervention_metrics,
)
from .interventions import (
    InterventionConfig, run_all_interventions,
)
from .hessian import compute_hessian_spectrum
from .repro import set_seed


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Per-layer pipeline
# ---------------------------------------------------------------------------

@dataclass
class LayerResults:
    layer: int
    accuracies: dict[str, float]                 # probe_name -> acc
    interventions: dict[str, dict]               # probe_name -> {INLP: {C,S,R,...}, ...}
    hessian: dict[str, dict]                     # probe_name -> spectrum dict
    mka_alignment: dict[str, float]              # probe_name -> MKA score X vs hidden
    val_zc_acc: float
    val_ze_acc: float

    def as_dict(self) -> dict:
        return {
            "layer": self.layer,
            "accuracies": self.accuracies,
            "interventions": self.interventions,
            "hessian": self.hessian,
            "mka_alignment": self.mka_alignment,
            "val_zc_acc": self.val_zc_acc,
            "val_ze_acc": self.val_ze_acc,
        }


def _train_three_probes(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    cfg: dict,
    device: torch.device,
) -> dict[str, Any]:
    dim = X_train.shape[1]
    probe_cfg = ProbeTrainConfig(
        epochs=cfg["probes"]["epochs"],
        lr=cfg["probes"]["lr"],
        weight_decay=cfg["probes"]["weight_decay"],
        batch_size=cfg["probes"]["batch_size"],
    )
    hidden = cfg["probes"]["hidden_dim"]

    linear = LinearProbe(dim).to(device)
    train_probe(linear, X_train, y_train, probe_cfg, device)

    mlp = MLPProbe(dim, hidden_dim=hidden).to(device)
    train_probe(mlp, X_train, y_train, probe_cfg, device)

    mka = MKAProbe(dim, hidden_dim=hidden,
                   mka_lambda=cfg["mka"]["lambda_reg"],
                   knn_k=cfg["mka"]["knn_k"]).to(device)
    train_probe(mka, X_train, y_train, probe_cfg, device)

    return {"linear": linear, "mlp": mlp, "mka": mka}


@torch.no_grad()
def _hidden_activations(probe, X: torch.Tensor,
                        device: torch.device,
                        batch_size: int = 1024) -> torch.Tensor:
    """Get the probe's hidden representation (post-ReLU for MLP/MKA, identity for Linear)."""
    probe.eval()
    out = []
    for start in range(0, X.shape[0], batch_size):
        xb = X[start:start + batch_size].to(device).float()
        _, h = probe(xb, return_hidden=True)
        out.append(h.cpu())
    return torch.cat(out, dim=0)


def _compute_mka_alignment(X: torch.Tensor, H: torch.Tensor,
                           knn_k: int = 10,
                           subsample: int = 1024) -> float:
    """MKA between original X and probe hidden H. Subsample to avoid OOM on big X."""
    n = X.shape[0]
    if n > subsample:
        idx = torch.randperm(n)[:subsample]
        X = X[idx]
        H = H[idx]
    K = knn_kernel(X, k=knn_k)
    L = knn_kernel(H, k=knn_k)
    return float(mka_score_fn(K, L).item())


def run_layer_pipeline(
    layer: int,
    X_probe: torch.Tensor, zc_probe_y: torch.Tensor, ze_probe_y: torch.Tensor,
    X_inter: torch.Tensor, zc_inter: torch.Tensor, ze_inter: torch.Tensor,
    X_test: torch.Tensor, zc_test: torch.Tensor, ze_test: torch.Tensor,
    cfg: dict,
    device: torch.device,
) -> LayerResults:
    """Run the full evaluation for one layer."""
    print(f"  [layer {layer}] training probes...")
    probes = _train_three_probes(X_probe, zc_probe_y, cfg, device)

    # Validation probes — trained on PROBE-TRAIN representations using clean labels
    print(f"  [layer {layer}] training validation probes...")
    val_cfg = ProbeTrainConfig(
        epochs=cfg["probes"]["epochs"],
        lr=cfg["probes"]["lr"],
        weight_decay=cfg["probes"]["weight_decay"],
        batch_size=cfg["probes"]["batch_size"],
    )
    val_probes = train_validation_probes(
        X_probe, zc_probe_y, ze_probe_y, val_cfg, device,
    )

    # Test-set accuracies of the three main probes
    accuracies = {}
    for name, probe in probes.items():
        accuracies[name] = probe_accuracy(probe, X_test, zc_test, device)

    # Interventions: applied to INTERVENTION-TRAIN reps, evaluated via val_probes
    print(f"  [layer {layer}] running interventions...")
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
    intervention_results: dict[str, dict] = {}
    # Same intervention X_post is used to evaluate every main probe variant
    # (the interventions act on the representation, not the probe).
    interventions_dict = run_all_interventions(
        X_inter, zc_inter, val_probes.zc_probe, device, inter_cfg,
    )
    for probe_name, probe in probes.items():
        intervention_results[probe_name] = {}
        for method, X_post in interventions_dict.items():
            metrics = compute_intervention_metrics(
                val_probes, X_pre=X_inter, X_post=X_post,
                zc=zc_inter, ze=ze_inter, device=device,
            )
            intervention_results[probe_name][method] = metrics.as_dict()

    # Hessian spectrum + directional alignment
    print(f"  [layer {layer}] computing Hessian spectra...")
    hessian_results: dict[str, dict] = {}
    for name, probe in probes.items():
        spectrum = compute_hessian_spectrum(
            probe, X_probe, zc_probe_y, device,
            top_n=cfg["hessian"]["top_n"],
            bottom_n=cfg["hessian"]["bottom_n"],
            max_iter=cfg["hessian"]["max_iter"],
            tol=cfg["hessian"]["tol"],
        )
        hessian_results[name] = spectrum.as_dict()

    # MKA alignment: between original X and probe's hidden representation
    print(f"  [layer {layer}] computing MKA alignment...")
    mka_alignment: dict[str, float] = {}
    for name, probe in probes.items():
        H = _hidden_activations(probe, X_probe, device)
        mka_alignment[name] = _compute_mka_alignment(
            X_probe, H, knn_k=cfg["mka"]["knn_k"],
        )

    return LayerResults(
        layer=layer,
        accuracies=accuracies,
        interventions=intervention_results,
        hessian=hessian_results,
        mka_alignment=mka_alignment,
        val_zc_acc=val_probes.acc_zc,
        val_ze_acc=val_probes.acc_ze,
    )


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_model_pipeline(cfg: dict, project_root: Path) -> dict:
    """Run the full pipeline for one model based on a loaded config."""
    set_seed(cfg["output"]["seed"])

    # 1. Data
    print("[data] loading + computing Ze...")
    paths = [project_root / p for p in cfg["data"]["paths"]]
    examples = build_examples(
        paths,
        max_examples=cfg["data"].get("max_examples"),
        seed=cfg["data"]["seed"],
    )
    train_ex, inter_ex, test_ex = split_balanced(
        examples,
        val_frac=cfg["data"]["val_frac"],
        inter_frac=cfg["data"]["inter_frac"],
        seed=cfg["data"]["seed"],
    )
    print(f"[data] probe-train={len(train_ex)}, inter={len(inter_ex)}, test={len(test_ex)}")

    # 2. Model
    print(f"[model] loading {cfg['model']['name']}...")
    device = pick_device()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[cfg["model"]["dtype"]]
    bundle = load_model(
        cfg["model"]["name"], device=device, dtype=dtype,
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    )

    # 3. Layer selection
    explicit_layers = cfg["extraction"].get("layers")
    if explicit_layers:
        layers = list(explicit_layers)
    else:
        layers = select_layers(
            bundle.n_layers,
            k=cfg["extraction"]["num_layers_to_probe"],
        )
    print(f"[layers] probing {layers}")

    # 4. Per-layer extraction + analysis
    cache_dir = project_root / cfg["output"]["cache_dir"]
    results_dir = project_root / cfg["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    all_layer_results: list[dict] = []

    for layer in layers:
        print(f"\n[layer {layer}] extracting representations...")
        X_probe, zc_probe, ze_probe = extract_layer_reps(
            bundle, train_ex, layer_idx=layer,
            batch_size=cfg["extraction"]["batch_size"],
            max_length=cfg["extraction"]["max_length"],
        )
        X_inter, zc_inter, ze_inter = extract_layer_reps(
            bundle, inter_ex, layer_idx=layer,
            batch_size=cfg["extraction"]["batch_size"],
            max_length=cfg["extraction"]["max_length"],
        )
        X_test, zc_test, ze_test = extract_layer_reps(
            bundle, test_ex, layer_idx=layer,
            batch_size=cfg["extraction"]["batch_size"],
            max_length=cfg["extraction"]["max_length"],
        )

        layer_results = run_layer_pipeline(
            layer,
            X_probe, zc_probe, ze_probe,
            X_inter, zc_inter, ze_inter,
            X_test, zc_test, ze_test,
            cfg, device,
        )
        all_layer_results.append(layer_results.as_dict())

        # Save incrementally so partial runs aren't lost
        import json
        out_path = results_dir / "results.json"
        with out_path.open("w") as f:
            json.dump({
                "model": cfg["model"]["name"],
                "layers": layers,
                "completed_layers": [r["layer"] for r in all_layer_results],
                "results": all_layer_results,
            }, f, indent=2)

    return {
        "model": cfg["model"]["name"],
        "layers": layers,
        "results": all_layer_results,
    }
