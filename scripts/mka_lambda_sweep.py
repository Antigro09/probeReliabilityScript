"""
Lambda sweep for the MKA regularizer (WS1 acceptance item e).

For each lambda in the grid, trains seed-matched (MLP, MKA) pairs and
reports, per lambda:
    - test accuracy of MKA vs the seed-matched MLP baseline
    - manifold alignment of the trained hidden layer (mka_score between
      the input's hard kNN kernel and the hidden RBF kernel, on held-out data)
    - parameter divergence from the seed-matched MLP

The regularizer "visibly shapes training" when alignment rises with lambda
and parameters diverge from the baseline; it "tanks accuracy" when test
accuracy drops materially below the MLP baseline. THE FINAL LAMBDA IS AN
EXPERIMENTER DECISION — record it in PREREGISTRATION_v2.md; this script
only produces the evidence table.

Data source is explicit — there is deliberately no fallback:
    --synthetic            clustered Gaussian binary data (no downloads; CI-safe)
    --config <yaml> --task <t> --layer <i>
                           real representations extracted from the config's
                           model (requires model download)

Usage:
    python -m scripts.mka_lambda_sweep --synthetic
    python -m scripts.mka_lambda_sweep --config configs/tiny.yaml --task sva
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.probes import (
    MLPProbe, MKAProbe, ProbeTrainConfig, train_probe, probe_accuracy,
    knn_kernel, rbf_kernel, mka_score,
)
from src.repro import set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true",
                     help="Use synthetic clustered data (no downloads)")
    src.add_argument("--config", help="Model config YAML for real representations")
    p.add_argument("--task", default="sva", choices=["sva", "gender", "sst2"],
                   help="Task (only with --config)")
    p.add_argument("--layer", type=int, default=None,
                   help="Layer to extract (only with --config; default: middle)")
    p.add_argument("--lambdas", default="0,0.01,0.03,0.1,0.3,1.0,3.0",
                   help="Comma-separated lambda grid")
    p.add_argument("--seeds", type=int, default=3, help="Seeds per lambda")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--n-synthetic", type=int, default=2000)
    p.add_argument("--d-synthetic", type=int, default=64)
    p.add_argument("--out", default=None,
                   help="Output JSONL (default results/mka_sweep/<tag>.jsonl)")
    return p.parse_args()


def synthetic_data(n: int, d: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    w = torch.randn(d, generator=g)
    y = (X @ w > 0).long()
    X = X + 0.5 * y.unsqueeze(1).float() * torch.randn(1, d, generator=g)
    return X, y


def real_data(config_path: str, task_name: str, layer: int | None):
    from src.pipeline import load_config
    from src.extraction import load_model, select_layers, extract_layer_reps, pick_device
    from src.tasks import get_task
    from scripts.run_benchmark import default_data_paths

    cfg = load_config(config_path if Path(config_path).is_absolute()
                      else PROJECT_ROOT / config_path)
    task = get_task(task_name)
    examples = task.load(default_data_paths(task_name, cfg),
                         max_examples=cfg["data"].get("max_examples"),
                         seed=cfg["data"]["seed"])
    device = pick_device()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[cfg["model"]["dtype"]]
    bundle = load_model(cfg["model"]["name"], device=device, dtype=dtype,
                        trust_remote_code=cfg["model"].get("trust_remote_code", False))
    layers = cfg["extraction"].get("layers") or select_layers(
        bundle.n_layers, k=cfg["extraction"]["num_layers_to_probe"])
    layer = layer if layer is not None else layers[len(layers) // 2]
    X, zc, _ = extract_layer_reps(bundle, examples, layer_idx=layer,
                                  batch_size=cfg["extraction"]["batch_size"],
                                  max_length=cfg["extraction"]["max_length"])
    return X, zc


@torch.no_grad()
def hidden_alignment(probe, X: torch.Tensor, k: int,
                     batch_size: int = 256) -> float:
    """Mean mka_score(knn(input), rbf(hidden)) over evaluation batches."""
    scores = []
    for start in range(0, X.shape[0], batch_size):
        xb = X[start:start + batch_size].float()
        if xb.shape[0] < k + 2:
            continue
        _, hidden = probe(xb, return_hidden=True)
        scores.append(mka_score(knn_kernel(xb, k=k),
                                rbf_kernel(hidden, k=k)).item())
    return statistics.mean(scores) if scores else float("nan")


def main():
    args = parse_args()
    device = torch.device("cpu") if args.synthetic else None
    if args.synthetic:
        X, y = synthetic_data(args.n_synthetic, args.d_synthetic)
        tag = "synthetic"
    else:
        from src.extraction import pick_device
        device = pick_device()
        X, y = real_data(args.config, args.task, args.layer)
        tag = f"{Path(args.config).stem}_{args.task}"
    device = device or torch.device("cpu")

    n = X.shape[0]
    n_train = int(0.7 * n)
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_te, y_te = X[n_train:], y[n_train:]
    lambdas = [float(x) for x in args.lambdas.split(",")]

    out_path = Path(args.out) if args.out else (
        PROJECT_ROOT / "results" / "mka_sweep" / f"{tag}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = ProbeTrainConfig(epochs=args.epochs, lr=1e-3, weight_decay=0.01,
                           batch_size=256)
    dim = X.shape[1]

    print(f"[sweep] data={tag} n={n} d={dim} lambdas={lambdas} "
          f"seeds={args.seeds} epochs={args.epochs}")
    header = (f"{'lambda':>8} {'acc_mka':>9} {'acc_mlp':>9} {'d_acc':>8} "
              f"{'align_mka':>10} {'align_mlp':>10} {'param_div':>10}")
    print(header)
    print("-" * len(header))

    rows = []
    with out_path.open("w") as f_out:
        for lam in lambdas:
            accs_mka, accs_mlp, aligns_mka, aligns_mlp, divs = [], [], [], [], []
            for s in range(args.seeds):
                seed = 1000 + s
                set_seed(seed)
                mlp = MLPProbe(dim, hidden_dim=args.hidden_dim)
                set_seed(seed)
                mka = MKAProbe(dim, hidden_dim=args.hidden_dim,
                               mka_lambda=lam, knn_k=args.knn_k)
                set_seed(seed + 1)
                train_probe(mlp, X_tr, y_tr, cfg, device)
                set_seed(seed + 1)
                t0 = time.time()
                train_probe(mka, X_tr, y_tr, cfg, device)
                wall = time.time() - t0

                accs_mka.append(probe_accuracy(mka, X_te, y_te, device))
                accs_mlp.append(probe_accuracy(mlp, X_te, y_te, device))
                aligns_mka.append(hidden_alignment(mka.cpu(), X_te, args.knn_k))
                aligns_mlp.append(hidden_alignment(mlp.cpu(), X_te, args.knn_k))
                divs.append(max((p1 - p2).abs().max().item()
                                for p1, p2 in zip(mlp.parameters(),
                                                  mka.parameters())))
                rows.append({
                    "lambda": lam, "seed": seed,
                    "acc_mka": accs_mka[-1], "acc_mlp": accs_mlp[-1],
                    "align_mka": aligns_mka[-1], "align_mlp": aligns_mlp[-1],
                    "param_divergence": divs[-1],
                    "wallclock_s": wall, "data": tag,
                    "epochs": args.epochs, "hidden_dim": args.hidden_dim,
                    "knn_k": args.knn_k,
                })
                f_out.write(json.dumps(rows[-1]) + "\n")

            print(f"{lam:>8.3g} {statistics.mean(accs_mka):>9.4f} "
                  f"{statistics.mean(accs_mlp):>9.4f} "
                  f"{statistics.mean(accs_mka) - statistics.mean(accs_mlp):>+8.4f} "
                  f"{statistics.mean(aligns_mka):>10.4f} "
                  f"{statistics.mean(aligns_mlp):>10.4f} "
                  f"{statistics.mean(divs):>10.3e}")

    print(f"\n[sweep] wrote {len(rows)} rows -> {out_path}")
    print("[sweep] Pick lambda where align_mka clearly exceeds align_mlp while "
          "d_acc stays small; record the choice in PREREGISTRATION_v2.md.")


if __name__ == "__main__":
    main()
