"""
End-to-end pipeline runner.

Usage:
    python -m scripts.run_model --config configs/pythia.yaml

Optional flags:
    --layers 1,6,12      Override layers in the config
    --max-examples 1000  Override data size for a quick test run
    --device cuda        Override device selection (cuda / cpu)
    --resume             Skip layers that already have results saved

Outputs (relative to project root):
    results/<model_name>/results.json     Per-layer results, written incrementally
    cache/<model_name>/<model>_L<layer>_n<N>.pt        Cached representations
    cache/<model_name>/<model>_L<layer>_n<N>.json     Provenance sidecar

The results.json schema is:
    {
        "model": "EleutherAI/pythia-160m",
        "layers": [1, 4, 6, 9, 12],
        "completed_layers": [...],
        "results": [
            {
                "layer": 1,
                "accuracies": {"linear": 0.7, "mlp": 0.74, "mka": 0.74},
                "interventions": {
                    "linear": {"INLP": {"C": ..., "S": ..., "R": ...}, ...},
                    "mlp": {...},
                    "mka": {...}
                },
                "hessian": {
                    "linear": {"lambda_max": ..., "mean_align_top": ...,
                               "eigenvalues_top": [...], "align_top": [...], ...},
                    ...
                },
                "mka_alignment": {"linear": ..., "mlp": ..., "mka": ...},
                "val_zc_acc": ...,
                "val_ze_acc": ...
            },
            ...
        ]
    }
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import load_config, run_model_pipeline


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True,
                   help="Path to YAML config file (e.g., configs/pythia.yaml)")
    p.add_argument("--layers", default=None,
                   help="Comma-separated layer indices to override config")
    p.add_argument("--max-examples", type=int, default=None,
                   help="Override config's max_examples")
    p.add_argument("--device", default=None, choices=["cuda", "cpu"],
                   help="Override device")
    return p.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)

    cfg = load_config(config_path)

    # CLI overrides
    if args.layers:
        cfg["extraction"]["layers"] = [int(x) for x in args.layers.split(",")]
        # explicit layers takes precedence over num_layers_to_probe
    if args.max_examples is not None:
        cfg["data"]["max_examples"] = args.max_examples

    # Show effective config
    print("=" * 70)
    print(f"  Model:    {cfg['model']['name']}")
    print(f"  Examples: {cfg['data'].get('max_examples', 'all')}")
    print(f"  Output:   {cfg['output']['results_dir']}")
    print("=" * 70)

    t0 = time.time()
    run_model_pipeline(cfg, project_root=PROJECT_ROOT)
    elapsed = time.time() - t0
    print(f"\n✅ Run complete in {elapsed/60:.1f} min")
    print(f"   Results: {PROJECT_ROOT / cfg['output']['results_dir'] / 'results.json'}")


if __name__ == "__main__":
    main()
