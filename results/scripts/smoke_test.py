"""
Smoke test: verify the full pipeline works end-to-end on a tiny slice.

Run from project root:
    python -m scripts.smoke_test

Stages:
    1. Data loading + Ze computation
    2. Balanced split sanity
    3. Device selection
    4. Model loading
    5. Tokenizer position validation
    6. Representation extraction
    7. Probe training (linear / mlp / mka)
    8. Intervention metrics computation
    9. Hessian directional alignment

Should take 2-4 minutes on a 5070. If this passes, run_model.py works.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from collections import Counter

from src.data import build_examples, split_balanced
from src.extraction import (
    load_model, select_layers, extract_layer_reps, pick_device,
    _validate_extraction_position,
)
from src.repro import set_seed
from src.probes import (
    LinearProbe, MLPProbe, MKAProbe,
    ProbeTrainConfig, train_probe, probe_accuracy,
)
from src.metrics import train_validation_probes, compute_intervention_metrics
from src.interventions import apply_fgsm
from src.hessian import compute_hessian_spectrum


def main():
    print("=" * 60)
    print("  FULL-PIPELINE SMOKE TEST")
    print("=" * 60)

    set_seed(42)

    data_dir = PROJECT_ROOT / "data"
    paths = [data_dir / "numpred.train",
             data_dir / "numpred.val"]
    for p in paths:
        if not p.exists():
            print(f"  ❌ Missing: {p}")
            sys.exit(1)

    # 1. Data
    print("\n[1/9] Loading 1500 examples + Ze computation...")
    examples = build_examples(paths, max_examples=1500, seed=42)
    zc_dist = Counter(ex.zc for ex in examples)
    ze_dist = Counter(ex.ze for ex in examples)
    print(f"      Zc: {dict(zc_dist)}    Ze: {dict(ze_dist)}")
    assert len(zc_dist) == 2 and len(ze_dist) == 2

    # 2. Split
    print("\n[2/9] Balanced split...")
    train, inter, test = split_balanced(examples)
    print(f"      probe-train={len(train)}  inter={len(inter)}  test={len(test)}")
    assert len(train) > 50, "split is too small for downstream stages"

    # 3. Device
    print("\n[3/9] Device...")
    device = pick_device()
    print(f"      Using: {device}")
    if device.type == "cuda":
        print(f"      GPU: {torch.cuda.get_device_name(0)}")

    # 4. Model
    print("\n[4/9] Loading Pythia-160M...")
    bundle = load_model("EleutherAI/pythia-160m", device=device)
    layer = select_layers(bundle.n_layers, k=3)[1]  # middle layer
    print(f"      Probing layer {layer}")

    # 5. Validation
    print("\n[5/9] Validating extraction position...")
    info = _validate_extraction_position(
        bundle, [ex.sentence for ex in train[:8]]
    )
    print(f"      Sample last tokens: {info['last_token_strings'][:3]}")

    # 6. Extraction (all three splits)
    print(f"\n[6/9] Extracting layer {layer} representations...")
    X_tr, zc_tr, ze_tr = extract_layer_reps(
        bundle, train, layer, batch_size=32,
        show_progress=False, validate=False,
    )
    X_in, zc_in, ze_in = extract_layer_reps(
        bundle, inter, layer, batch_size=32,
        show_progress=False, validate=False,
    )
    X_te, zc_te, _ = extract_layer_reps(
        bundle, test, layer, batch_size=32,
        show_progress=False, validate=False,
    )
    print(f"      X_train.shape={tuple(X_tr.shape)} "
          f"X_inter.shape={tuple(X_in.shape)} "
          f"X_test.shape={tuple(X_te.shape)}")

    # 7. Probe training
    print("\n[7/9] Training Linear / MLP / MKA probes (5 epochs)...")
    cfg = ProbeTrainConfig(epochs=5, lr=1e-3, weight_decay=0.01,
                           batch_size=64)
    dim = X_tr.shape[1]
    linear = LinearProbe(dim).to(device)
    train_probe(linear, X_tr, zc_tr, cfg, device)
    mlp = MLPProbe(dim, hidden_dim=128).to(device)
    train_probe(mlp, X_tr, zc_tr, cfg, device)
    mka = MKAProbe(dim, hidden_dim=128, mka_lambda=0.1, knn_k=5).to(device)
    train_probe(mka, X_tr, zc_tr, cfg, device)

    acc_lin = probe_accuracy(linear, X_te, zc_te, device)
    acc_mlp = probe_accuracy(mlp, X_te, zc_te, device)
    acc_mka = probe_accuracy(mka, X_te, zc_te, device)
    print(f"      Test accuracy — Linear: {acc_lin:.3f}  "
          f"MLP: {acc_mlp:.3f}  MKA: {acc_mka:.3f}")

    # 8. Validation probes + one intervention
    print("\n[8/9] Validation probes + FGSM intervention...")
    val_cfg = ProbeTrainConfig(epochs=5, lr=1e-3, weight_decay=0.01,
                               batch_size=64)
    val_probes = train_validation_probes(X_tr, zc_tr, ze_tr, val_cfg, device,
                                         min_acc=0.0)
    print(f"      Validation Zc acc={val_probes.acc_zc:.3f}  "
          f"Ze acc={val_probes.acc_ze:.3f}")
    X_post = apply_fgsm(X_in, zc_in, validation_probe=val_probes.zc_probe,
                        device=device, epsilon=0.5)
    metrics = compute_intervention_metrics(
        val_probes, X_pre=X_in, X_post=X_post, zc=zc_in, ze=ze_in,
        device=device,
    )
    print(f"      FGSM:  C={metrics.completeness:.3f}  "
          f"S={metrics.selectivity:.3f}  R={metrics.reliability:.3f}")

    # 9. Hessian
    print("\n[9/9] Hessian spectrum (top-5) for Linear probe...")
    spec = compute_hessian_spectrum(
        linear, X_tr, zc_tr, device,
        top_n=5, bottom_n=5, max_iter=20, tol=1e-2,
    )
    print(f"      lambda_max={spec.lambda_max:.4f}  "
          f"mean_align_top={spec.mean_align_top:.3f}")
    assert spec.lambda_max > 0, "lambda_max should be positive"
    assert 0 <= spec.mean_align_top <= 1, "alignment must be in [0, 1]"

    print("\n" + "=" * 60)
    print("  ✅ ALL STAGES PASSED")
    print("=" * 60)
    print("\nNext step: run a tiny end-to-end run with")
    print("  python -m scripts.run_model --config configs/tiny.yaml")


if __name__ == "__main__":
    main()
