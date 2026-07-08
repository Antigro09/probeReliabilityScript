"""
WS5 driver — synthetic construct-validity study for the repaired metric R_rep.

Runs the environment-shift synthetic study (src/ws5_synthetic.py) through the
repaired-reliability machinery (src/ws5_repaired.py), producing:

  1. per-candidate JSONL rows: ground-truth recovery rho_rec, contamination
     kappa, the alignment predictor A (for the downstream real-data claim — NOT
     used to define R_rep), and the full decomposition ladder
     (R_v1_max, R_v1_mean, R_excl, R_cand);
  2. results/ws5/battery.json — the 5 registered validation checks (i)-(v),
     including the confident-shortcut DISSOCIATION check that answers ydqQ;
  3. results/ws5/headtohead.json — Kendall tau of each ladder rung against
     rho_rec, with a replicate-clustered bootstrap CI on the R_rep-vs-v1 gap.

Design & the meaning of every knob: docs/WS5_DESIGN.md. All thresholds below are
PLACEHOLDERS to lock in PREREGISTRATION_v2.md.

Reproducing the v1 defect on purpose: the five v1-method edits and the two
concept edits (INLP/RLACE) are candidate-INDEPENDENT, so R_v1_* and R_excl are
constant across candidates within a replicate — exactly why v1 R cannot rank
probes. Only R_cand varies per candidate.

Usage:
    python -m scripts.ws5_run --family linear --replicates 5 --seeds 4
    python -m scripts.ws5_run --family nonlinear --rank 2
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.probes import (
    LinearProbe, MLPProbe, MKAProbe, ProbeTrainConfig, train_probe, probe_accuracy,
)
from src.metrics import train_validation_probes, DECODABILITY_FLOOR
from src.interventions import apply_alterrep, apply_fgsm, apply_pgd
from src.hessian import compute_hessian_spectrum
from src.repro import set_seed
from src.gates import learnability_gate
from src.ws5_synthetic import (
    SyntheticConfig, make_planted_reps, split_eval_folds,
    recovery_score, subspace_recovery, random_cosine_baseline, check_4cell_balance,
)
from src.ws5_repaired import (
    candidate_direction, candidate_edit, train_independent_evaluators,
    repaired_reliability, concept_edits_once, EvaluatorQualityError, EVAL_SEED_BASE,
)

SCHEMA_VERSION = 2
ARCHS = ("linear", "mlp", "mka")

# ---- Registered battery thresholds (PLACEHOLDERS — lock in PREREGISTRATION_v2.md) ----
ICC_MIN = 0.70          # (i) estimator can tell candidates apart
SAT_DELTA = 0.02        # (ii) saturation band
SAT_MAX = 0.20
RANGE_MIN = 0.30
SHUF_MAX = 0.10         # (iii) shuffled-candidate ceiling (P95)
RAND_MARGIN_MIN = 0.30  # (iv) good candidate must beat random by this
SHORT_MAX = 0.10        # (v) confident-shortcut R_rep ceiling
TAU_MIN = 0.50          # head-to-head P1
DELTA_MIN = 0.15        # head-to-head P2


def make_candidate(arch, dim, cfg, seed):
    g = torch.Generator().manual_seed(seed)
    if arch == "linear":
        probe = LinearProbe(dim)
    elif arch == "mlp":
        probe = MLPProbe(dim, hidden_dim=cfg["hidden_dim"])
    elif arch == "mka":
        probe = MKAProbe(dim, hidden_dim=cfg["hidden_dim"],
                         mka_lambda=cfg["mka_lambda"], knn_k=cfg["knn_k"])
    else:
        raise ValueError(arch)
    for p in probe.parameters():
        if p.dim() >= 2:
            torch.nn.init.kaiming_uniform_(p, a=5 ** 0.5, generator=g)
        else:
            torch.nn.init.zeros_(p)
    return probe


def contamination(d_c, v_s) -> float:
    return recovery_score(d_c, v_s)


def run_replicate(family, alpha, replicate, args, device, train_cfg, probe_cfg, floor):
    """Return (rows, meta) for one generator replicate."""
    scfg = SyntheticConfig(
        D=args.D, N=args.N, family=family, alpha_train=alpha,
        alpha_eval=0.5, vs_vc_corr=args.vs_vc_corr, seed=1000 * replicate + int(alpha * 100),
    )
    reps_train, reps_eval, truth = make_planted_reps(scfg)
    try:
        check_4cell_balance(reps_train, tol=0.25)
    except ValueError as e:
        print(f"[warn] replicate {replicate} alpha={alpha}: {e}")

    folds = split_eval_folds(reps_eval, ref_frac=0.3, eval_frac=0.35,
                             seed=7919 * replicate)
    X_ref = folds["ref"]["X"]
    X_eval, zc_eval, ze_eval = folds["eval"]["X"], folds["eval"]["zc"], folds["eval"]["ze"]
    X_inter, zc_inter, ze_inter = folds["inter"]["X"], folds["inter"]["zc"], folds["inter"]["ze"]

    # WS2 learnability gate on the eval env (must decode the concept there).
    # SKIP for the non-linear (XOR) family: that concept is intentionally NOT
    # linearly decodable, so the linear gate probe would always fail (~chance).
    # Decodability is instead enforced by the MLP-certifier quality gate
    # (EVAL_MIN_ACC) in train_independent_evaluators below.
    if family != "nonlinear":
        gate = learnability_gate(X_eval, zc_eval, ze_eval, device, train_cfg,
                                 zc_floor=0.60, ze_floor=0.55, chance=0.5)
        if not gate.passed:
            print(f"[gate] replicate {replicate} alpha={alpha} FAILED: {gate.report()}")
            return [], {"gate_passed": False}

    # Certifiers (bagged, once per replicate)
    try:
        evaluators = train_independent_evaluators(
            X_eval, zc_eval, ze_eval, train_cfg, device,
            n_evaluators=args.evaluators, seed_base=EVAL_SEED_BASE + 37 * replicate,
            # A linear certifier cannot read an XOR concept -> MLP certifiers for
            # the non-linear family (WS5_DESIGN §6.2). rho_rec uses the planted
            # subspace, so the candidate direction is still compared to truth.
            certifier_arch=("mlp" if family == "nonlinear" else "linear"),
        )
    except EvaluatorQualityError as e:
        print(f"[eval] replicate {replicate} alpha={alpha} invalid: {e}")
        return [], {"gate_passed": True, "evaluators_ok": False}

    # Candidate-INDEPENDENT edits, once per replicate.
    concept_edits = concept_edits_once(X_inter, zc_inter, device,
                                       inlp_iters=10, rlace_rank=1, rlace_steps=args.rlace_steps)
    # v1-style validation probe supplies the 3 self-referential v1 edits.
    set_seed(EVAL_SEED_BASE - 1)
    v1_vp = train_validation_probes(reps_train["X"], reps_train["zc"], reps_train["ze"],
                                    train_cfg, device, min_acc=0.0)
    extra_edits = {
        "AlterRep": apply_alterrep(X_inter, zc_inter, validation_probe=v1_vp.zc_probe,
                                   device=device, alpha=1.0),
        "FGSM": apply_fgsm(X_inter, zc_inter, validation_probe=v1_vp.zc_probe,
                           device=device, epsilon=0.5),
        "PGD": apply_pgd(X_inter, zc_inter, validation_probe=v1_vp.zc_probe,
                         device=device, epsilon=0.5),
    }

    V = truth["concept_subspace"]      # (D, m)
    v_s = truth["v_s"]
    rows = []

    def score_candidate(arch, seed, shuffled=False, random_dir=False):
        # candidate training targets (optionally shuffled labels)
        zc_train = reps_train["zc"]
        if shuffled:
            g = torch.Generator().manual_seed(424242 + seed)
            zc_train = zc_train[torch.randperm(zc_train.shape[0], generator=g)]
        probe = make_candidate(arch, args.D, probe_cfg, seed).to(device)
        train_probe(probe, reps_train["X"], zc_train, train_cfg, device)
        acc = probe_accuracy(probe, X_inter, zc_inter, device)   # eval-env accuracy (invariant)
        acc_train = probe_accuracy(probe, reps_train["X"], reps_train["zc"], device)

        # A on the candidate's TRAINING-loss Hessian (matches the real pipeline)
        spec = compute_hessian_spectrum(probe, reps_train["X"], zc_train, device,
                                        top_n=20, bottom_n=0, max_iter=50, tol=1e-3)
        A = spec.mean_align_top

        d_override = None
        if random_dir:
            gg = torch.Generator().manual_seed(9001 + seed)
            d_override = torch.randn(args.D, generator=gg)

        Q = None if random_dir else candidate_direction(probe, X_ref, device, r=args.rank)
        if Q is not None:
            # subspace_recovery handles both the 1-D (linear) and 2-D (nonlinear)
            # planted concept uniformly; for (D,1) vs (D,1) it reduces to |cos|.
            rho_rec = subspace_recovery(Q, V)
            kappa = contamination(Q[:, 0], v_s)
        else:
            rho_rec, kappa = None, None

        rr = repaired_reliability(
            probe, X_ref=X_ref, X_inter=X_inter, zc_inter=zc_inter, ze_inter=ze_inter,
            evaluators=evaluators, device=device, floor=floor, rank=args.rank,
            d_override=d_override, concept_edits=concept_edits, extra_edits=extra_edits,
        )
        row = {
            "schema_version": SCHEMA_VERSION, "family": family, "alpha_train": alpha,
            "replicate": replicate, "arch": arch, "seed": seed,
            "control": ("shuffled" if shuffled else "random" if random_dir else "none"),
            "accuracy_eval": acc, "accuracy_train": acc_train,
            "A": A, "lambda_max": spec.lambda_max,
            "rho_rec": rho_rec, "kappa": kappa,
            "rand_cos_baseline": random_cosine_baseline(args.D),
            **rr,
        }
        return row

    for arch in ARCHS:
        for k in range(args.seeds):
            rows.append(score_candidate(arch, seed=1000 + k))

    # Controls (WS5_DESIGN §4 iii/iv): shuffled-label + random-direction pseudo-candidates
    if args.controls:
        for k in range(args.control_draws):
            rows.append(score_candidate("mlp", seed=5000 + k, shuffled=True))
            rows.append(score_candidate("mlp", seed=6000 + k, random_dir=True))

    return rows, {"gate_passed": True, "evaluators_ok": True}


# ---------------------------------------------------------------------------
# Analysis: battery + head-to-head
# ---------------------------------------------------------------------------

def _defined(rows, key, control="none"):
    return [r for r in rows if r.get("control") == control and r.get(key) is not None]


def battery(rows, D) -> dict:
    import numpy as np
    out = {"thresholds": {
        "ICC_MIN": ICC_MIN, "SAT_MAX": SAT_MAX, "RANGE_MIN": RANGE_MIN,
        "SHUF_MAX": SHUF_MAX, "RAND_MARGIN_MIN": RAND_MARGIN_MIN, "SHORT_MAX": SHORT_MAX,
    }}
    main = _defined(rows, "R_cand", "none")

    # (i) ICC: between-candidate variance vs within-candidate certifier SE
    Rc = np.array([r["R_cand"] for r in main], dtype=float)
    ses = np.array([(r["R_rep_sd"].get("cand") or 0.0) / max(1, r["n_evaluators"]) ** 0.5
                    for r in main], dtype=float)
    var_between = float(np.var(Rc)) if len(Rc) > 1 else 0.0
    mean_se2 = float(np.mean(ses ** 2)) if len(ses) else 0.0
    icc = var_between / (var_between + mean_se2) if (var_between + mean_se2) > 0 else 0.0
    out["i_reliability"] = {"icc": icc, "pass": icc >= ICC_MIN,
                            "var_between": var_between, "mean_se2": mean_se2}

    # (ii) saturation + dynamic range over the top rho_rec band
    if len(Rc) >= 10:
        rho = np.array([r["rho_rec"] for r in main if r["rho_rec"] is not None])
        top = Rc[np.argsort([r["rho_rec"] or 0 for r in main])[-len(Rc) // 3:]]
        sat_frac = float(np.mean(top >= 1 - SAT_DELTA))
        rng = float(np.percentile(Rc, 95) - np.percentile(Rc, 5))
        out["ii_saturation"] = {"sat_frac": sat_frac, "range": rng,
                                "pass": (sat_frac <= SAT_MAX) and (rng >= RANGE_MIN)}
    else:
        out["ii_saturation"] = {"pass": None, "note": "too few defined candidates"}

    # (iii) shuffled-candidate collapse
    shuf = [r["R_cand"] for r in _defined(rows, "R_cand", "shuffled")]
    if shuf:
        p95 = float(np.percentile(shuf, 95))
        out["iii_shuffled"] = {"p95": p95, "pass": p95 <= SHUF_MAX, "n": len(shuf)}
    else:
        out["iii_shuffled"] = {"pass": None, "note": "no shuffled controls (run with --controls)"}

    # (iv) random-direction floor + margin
    rand = [r["R_cand"] for r in _defined(rows, "R_cand", "random")]
    if rand and len(Rc):
        p95r = float(np.percentile(rand, 95))
        good = float(np.percentile(Rc, 90))
        out["iv_random"] = {"p95_random": p95r, "good_p90": good,
                            "margin": good - p95r, "pass": (good - p95r) >= RAND_MARGIN_MIN}
    else:
        out["iv_random"] = {"pass": None, "note": "no random controls (run with --controls)"}

    # (v) confident-shortcut DISSOCIATION — the crux of ydqQ
    if len(main) >= 10:
        A = np.array([r["A"] for r in main], dtype=float)
        a_hi = float(np.percentile(A, 75))
        # high-A, low-R_cand cell must be populated
        hiA_loR = [(r["A"], r["R_cand"]) for r in main
                   if r["A"] >= a_hi and r["R_cand"] <= SHORT_MAX]
        quads = {
            "hiA_hiR": sum(1 for r in main if r["A"] >= a_hi and r["R_cand"] > SHORT_MAX),
            "hiA_loR": len(hiA_loR),
            "loA_hiR": sum(1 for r in main if r["A"] < a_hi and r["R_cand"] > SHORT_MAX),
            "loA_loR": sum(1 for r in main if r["A"] < a_hi and r["R_cand"] <= SHORT_MAX),
        }
        out["v_dissociation"] = {
            "A_high_threshold": a_hi, "quadrants": quads,
            "pass": quads["hiA_loR"] > 0,   # dissociation exists
            "note": "PASS requires a populated high-A / low-R_rep cell",
        }
    else:
        out["v_dissociation"] = {"pass": None, "note": "too few candidates"}

    out["all_pass"] = all(v.get("pass") for k, v in out.items()
                          if isinstance(v, dict) and "pass" in v and v.get("pass") is not None)
    return out


def head_to_head(rows) -> dict:
    import numpy as np
    from scipy.stats import kendalltau
    main = [r for r in rows if r.get("control") == "none" and r.get("rho_rec") is not None]
    if len(main) < 8:
        return {"note": "too few candidates for head-to-head"}

    rho_rec = np.array([r["rho_rec"] for r in main])
    reps = np.array([r["replicate"] for r in main])

    def tau_vs_truth(key):
        vals = np.array([(r[key] if r[key] is not None else np.nan) for r in main])
        mask = ~np.isnan(vals)
        if mask.sum() < 8:
            return float("nan")
        t, _ = kendalltau(vals[mask], rho_rec[mask])
        return float(t)

    taus = {rung: tau_vs_truth(rung)
            for rung in ("R_v1_max", "R_v1_mean", "R_excl", "R_cand")}

    # Replicate-clustered bootstrap on Delta tau = tau(R_cand) - tau(R_v1_max)
    rng = np.random.default_rng(0)
    uniq = np.unique(reps)
    deltas = []
    for _ in range(2000):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(reps == r)[0] for r in pick])
        rr = rho_rec[idx]
        rc = np.array([main[i]["R_cand"] if main[i]["R_cand"] is not None else np.nan for i in idx])
        rv = np.array([main[i]["R_v1_max"] if main[i]["R_v1_max"] is not None else np.nan for i in idx])
        m = ~(np.isnan(rc) | np.isnan(rv))
        if m.sum() < 8:
            continue
        tc, _ = kendalltau(rc[m], rr[m])
        tv, _ = kendalltau(rv[m], rr[m])
        deltas.append(tc - tv)
    deltas = np.array(deltas)
    ci = (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))) if len(deltas) else (float("nan"),) * 2

    return {
        "tau": taus, "delta_tau_R_cand_minus_v1max": taus["R_cand"] - taus["R_v1_max"],
        "delta_tau_ci95": ci,
        "P1_tau_rep_ge_min": (taus["R_cand"] >= TAU_MIN),
        "P2_delta_ci_lower_gt_0": (ci[0] > 0),
        "thresholds": {"TAU_MIN": TAU_MIN, "DELTA_MIN": DELTA_MIN},
        "n": len(main),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="linear", choices=["linear", "nonlinear"])
    ap.add_argument("--alphas", default="0.5,0.7,0.9,0.98",
                    help="comma-separated alpha_train grid")
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=4, help="candidate seeds per (arch,alpha,replicate)")
    ap.add_argument("--evaluators", type=int, default=5)
    ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--N", type=int, default=8000)
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--vs-vc-corr", type=float, default=0.0)
    ap.add_argument("--rlace-steps", type=int, default=200)
    ap.add_argument("--controls", action="store_true", help="also run shuffled + random pseudo-candidates")
    ap.add_argument("--control-draws", type=int, default=20)
    ap.add_argument("--floor", type=float, default=DECODABILITY_FLOOR)
    ap.add_argument("--out-dir", default="results/ws5")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg = ProbeTrainConfig(epochs=50, lr=1e-3, weight_decay=0.01, batch_size=256)
    probe_cfg = {"hidden_dim": 256, "mka_lambda": 0.5, "knn_k": 10}
    alphas = [float(a) for a in args.alphas.split(",")]

    out_dir = (Path(args.out_dir) if Path(args.out_dir).is_absolute()
               else PROJECT_ROOT / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"ws5_{args.family}.jsonl"

    all_rows = []
    with rows_path.open("w") as f:
        for rep in range(args.replicates):
            for alpha in alphas:
                print(f"[run] family={args.family} replicate={rep} alpha={alpha}")
                rows, meta = run_replicate(args.family, alpha, rep, args, device,
                                           train_cfg, probe_cfg, args.floor)
                for r in rows:
                    f.write(json.dumps(r) + "\n")
                all_rows.extend(rows)

    bat = battery(all_rows, args.D)
    h2h = head_to_head(all_rows)
    (out_dir / f"battery_{args.family}.json").write_text(json.dumps(bat, indent=2))
    (out_dir / f"headtohead_{args.family}.json").write_text(json.dumps(h2h, indent=2))

    print("\n=== BATTERY ===")
    print(json.dumps(bat, indent=2))
    print("\n=== HEAD-TO-HEAD ===")
    print(json.dumps(h2h, indent=2))
    print(f"\nRows: {rows_path}")


if __name__ == "__main__":
    main()
