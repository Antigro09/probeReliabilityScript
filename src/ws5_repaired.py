"""
WS5 — Repaired reliability metric (R_rep).

See docs/WS5_DESIGN.md for the full rationale. In one paragraph: the v1
benchmark computes reliability R once per layer from validation + data-derived
probes, so R is byte-identical across every architecture and seed in a cell and
therefore cannot be what a per-probe predictor A ranks. R_rep instead ties the
edit to the *candidate probe's own extracted direction*, certifies it with
*data-independent, bagged evaluator probes*, applies the WS3 decodability floor,
and aggregates by MEAN over a decomposition ladder — never `max`. Whether the
resulting A<->R_rep link is non-circular is decided by the synthetic study
(src/ws5_synthetic.py) + the dissociation battery, NOT by this module.

This module builds R_rep entirely by composing the existing public APIs of
src/{probes,metrics,interventions,repro}.py. It does not modify them, and it
never calls AlterRep/FGSM/PGD to define R_rep (those re-inject the probe's own
params/gradient and re-open the F1 circularity channel — WS5_DESIGN §2.2).

Key design corrections baked in (from the adversarial review of the first draft;
WS5_DESIGN §2-3):
  * Direction is the top-r right-singular vector(s) of the per-example
    logit-difference Jacobian over a FIXED reference set, so estimation noise is
    matched across architectures (the raw mean-gradient is exact for Linear but
    noisy for MLP/MKA, which would bias recovery pro-linear). For a Linear probe
    the Jacobian is rank-1 and this returns exactly normalize(w[1]-w[0]).
  * F1 is NOT claimed severed for linear R_cand; the non-circularity rests on the
    synthetic dissociation. `d_override` exists so the C-a(E2) control can score
    an edit whose direction is provably independent of the candidate's weights.
  * Evaluator independence comes from DATA (bagging the eval fold), not from the
    RNG seed alone (convex linear probes trained on the same fold converge to the
    same discriminant regardless of seed).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .probes import ProbeTrainConfig, probe_accuracy, train_probe, LinearProbe, MLPProbe
from .metrics import (
    ValidationProbes,
    train_validation_probes,
    compute_intervention_metrics,
    DECODABILITY_FLOOR,
)
from .interventions import apply_inlp, apply_rlace
from .repro import set_seed


# --- Constants; the *_MIN / *_FRAC values are experimenter knobs (WS5_DESIGN §8) ---
EVAL_SEED_BASE = 900_000       # fixed, disjoint from candidate seeds (1000+k) and gate seed (777)
EVAL_MIN_ACC = 0.60            # D10: out-of-sample evaluator floor; below -> cell invalid.
#                                Anchored to the learnability gate's zc floor (0.60): if the
#                                task was declared learnable at 0.60 on the candidate fold, a
#                                certifier on the eval fold must also reach ~0.60, else it is
#                                BLIND and the cell is invalid. Set here (not higher) because
#                                small models decode sentiment/gender at only ~0.65-0.75 at the
#                                last token, and the WS3 per-metric floor (0.55) already nulls
#                                degenerate C/S — this gate only rules out a broken certifier.
BAG_FRAC = 0.8                 # D1: bootstrap size as a fraction of the eval fold
GRAD_NORM_MIN = 1e-6           # below this RMS gradient magnitude the direction is degenerate -> None
CLONE_COS_MAX = 1.0 - 1e-4     # evaluators more aligned than this are near-clones (bagging failed)


class EvaluatorQualityError(RuntimeError):
    """Raised when a certifier cannot decode the concept from CLEAN reps.

    This is a *cell-invalid* condition (the evaluator is blind), deliberately
    distinct from a candidate legitimately erasing the concept (which yields a
    low but VALID R). The driver should mark the cell invalid, not score it 0.
    """


# ---------------------------------------------------------------------------
# 2.1  Architecture-neutral candidate direction
# ---------------------------------------------------------------------------

def candidate_direction(
    probe,
    X_ref: torch.Tensor,
    device: torch.device,
    r: int = 1,
    batch_size: int = 512,
    grad_norm_min: float = GRAD_NORM_MIN,
) -> torch.Tensor | None:
    """Top-`r` right-singular vectors of the per-example logit-difference
    Jacobian of `probe` over the fixed reference set `X_ref`.

    Returns an orthonormal `(D, r)` basis `Q` (columns), or `None` if the
    gradient field is degenerate (probe is ~flat). For a `LinearProbe` the
    Jacobian rows are all equal to `w[1]-w[0]`, so `Q[:,0]` == normalize(w[1]-w[0]).

    Grad-enabled forward via `probe(xb)` — NOT `probe_logits` (which is
    @torch.no_grad, src/probes.py:328).
    """
    probe.eval()
    n, d = X_ref.shape
    G = torch.empty(n, d, dtype=torch.float32)
    for s in range(0, n, batch_size):
        xb = X_ref[s:s + batch_size].clone().float().to(device).requires_grad_(True)
        logits = probe(xb)
        if logits.shape[1] != 2:
            raise AssertionError(
                "WS5 R_rep is binary-only: it uses the logit-difference "
                f"(logit_1 - logit_0), but probe emitted {logits.shape[1]} classes."
            )
        diff = logits[:, 1] - logits[:, 0]
        # d(sum_i diff_i)/d xb == per-example gradients, since diff_i depends only on xb_i.
        g = torch.autograd.grad(diff.sum(), xb)[0]
        G[s:s + xb.shape[0]] = g.detach().float().cpu()

    # RMS gradient magnitude — degeneracy guard independent of the SVD direction.
    g_rms = (G.pow(2).sum(dim=1).mean().sqrt()).item()
    if g_rms < grad_norm_min:
        return None

    # Top-r right-singular vectors (directions in R^D capturing the dominant
    # gradient axis). Vh rows are orthonormal; QR is a defensive re-orthonormalize.
    _, _, Vh = torch.linalg.svd(G, full_matrices=False)
    r_eff = max(1, min(r, Vh.shape[0]))
    D_c = Vh[:r_eff].T.contiguous()          # (D, r_eff)
    Q, _ = torch.linalg.qr(D_c)
    return Q


def candidate_projection(Q: torch.Tensor) -> torch.Tensor:
    """Population-level erasure projector `I - Q Qᵀ` for an orthonormal `(D, r)`
    basis `Q`. Rank-`r` by construction."""
    d = Q.shape[0]
    return torch.eye(d, dtype=Q.dtype) - Q @ Q.T


def candidate_edit(
    probe,
    X_ref: torch.Tensor,
    X_inter: torch.Tensor,
    device: torch.device,
    r: int = 1,
    d_override: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Produce the candidate-derived intervened reps `X_inter @ (I - Q Qᵀ)`.

    If `d_override` (shape `(D,)` or `(D, r)`) is given, it is used as the
    erasure basis instead of the candidate's own direction — this is how the
    C-a(E2) falsification control scores an edit whose direction is provably
    independent of the candidate's weights (WS5_DESIGN §3.2 C-a).

    Returns `None` if the candidate direction is degenerate.
    """
    if d_override is not None:
        Q = d_override.float()
        if Q.dim() == 1:
            Q = Q.unsqueeze(1)
        Q, _ = torch.linalg.qr(Q)
    else:
        Q = candidate_direction(probe, X_ref, device, r=r)
        if Q is None:
            return None
    P_erase = candidate_projection(Q)
    return (X_inter.float() @ P_erase).contiguous()


# ---------------------------------------------------------------------------
# 2.3  Independent, bagged, quality-gated certifiers
# ---------------------------------------------------------------------------

@dataclass
class Evaluator:
    """A certifier probe pair plus its out-of-sample quality (WS5_DESIGN §2.3).

    `train_validation_probes` reports IN-SAMPLE accuracy (src/metrics.py:104);
    we recompute accuracy on the bootstrap out-of-bag slice so the quality gate
    is honest.
    """
    probes: ValidationProbes
    seed: int
    acc_zc_oos: float
    acc_ze_oos: float


def _train_certifier_pair(X, zc, ze, cfg, device, arch: str) -> ValidationProbes:
    """Train a (zc, ze) certifier pair of the given arch on X. 'linear' matches
    metrics.train_validation_probes; 'mlp' is needed for the non-linear
    synthetic family (a linear certifier cannot read an XOR concept — WS5_DESIGN
    §6.2, C-b'). Accuracies are in-sample here; the caller recomputes OOB."""
    if arch == "linear":
        return train_validation_probes(X, zc, ze, cfg, device, min_acc=0.0)
    if arch != "mlp":
        raise ValueError(f"certifier_arch must be 'linear' or 'mlp', got {arch!r}")
    dim = X.shape[1]
    zc_p = MLPProbe(dim).to(device)
    train_probe(zc_p, X, zc, cfg, device)
    ze_p = MLPProbe(dim).to(device)
    train_probe(ze_p, X, ze, cfg, device)
    return ValidationProbes(zc_p, ze_p,
                            probe_accuracy(zc_p, X, zc, device),
                            probe_accuracy(ze_p, X, ze, device))


def train_independent_evaluators(
    X_eval: torch.Tensor,
    zc_eval: torch.Tensor,
    ze_eval: torch.Tensor,
    cfg: ProbeTrainConfig,
    device: torch.device,
    n_evaluators: int = 5,
    bag_frac: float = BAG_FRAC,
    min_acc: float = EVAL_MIN_ACC,
    seed_base: int = EVAL_SEED_BASE,
    clone_cos_max: float = CLONE_COS_MAX,
    certifier_arch: str = "linear",
) -> list[Evaluator]:
    """Train `n_evaluators` certifier pairs, each on an independent bootstrap
    resample (bagging) of the clean evaluator fold `X_eval`.

    Independence comes from resampling the DATA, not just reseeding — convex
    linear probes on the same fold converge to the same discriminant regardless
    of seed. Each evaluator's out-of-bag accuracy must clear `min_acc`, else the
    cell is invalid (EvaluatorQualityError). A near-clone check catches a failed
    bag.
    """
    n = X_eval.shape[0]
    if n < 8:
        raise ValueError(f"evaluator fold needs >= 8 examples, got {n}")
    n_bag = max(2, int(bag_frac * n))
    evaluators: list[Evaluator] = []

    for e in range(n_evaluators):
        seed = seed_base + e
        set_seed(seed)
        g = torch.Generator().manual_seed(seed)
        bag_idx = torch.randint(0, n, (n_bag,), generator=g)
        mask = torch.ones(n, dtype=torch.bool)
        mask[bag_idx.unique()] = False
        oob_idx = mask.nonzero(as_tuple=True)[0]
        if oob_idx.numel() < 4:
            # Degenerate bag (almost no out-of-bag); widen deterministically.
            oob_idx = torch.arange(n)[::5]

        vp = _train_certifier_pair(
            X_eval[bag_idx], zc_eval[bag_idx], ze_eval[bag_idx],
            cfg, device, certifier_arch,   # OOB accuracy gate below, not the print-only warn
        )
        acc_zc_oos = probe_accuracy(vp.zc_probe, X_eval[oob_idx], zc_eval[oob_idx], device)
        acc_ze_oos = probe_accuracy(vp.ze_probe, X_eval[oob_idx], ze_eval[oob_idx], device)
        if acc_zc_oos < min_acc or acc_ze_oos < min_acc:
            raise EvaluatorQualityError(
                f"evaluator seed={seed} out-of-sample acc below floor "
                f"(zc={acc_zc_oos:.3f}, ze={acc_ze_oos:.3f}, floor={min_acc}); "
                f"the certifier is blind — cell is invalid, not low-R."
            )
        evaluators.append(Evaluator(vp, seed, acc_zc_oos, acc_ze_oos))

    # Clone check only meaningful for linear certifiers (a discriminant
    # direction). MLP certifiers get their independence from bagging; skip.
    if certifier_arch == "linear":
        _assert_not_clones(evaluators, clone_cos_max)
    return evaluators


def _discriminant(vp: ValidationProbes) -> torch.Tensor:
    z = vp.zc_probe
    if hasattr(z, "linear"):
        w = z.linear.weight.detach().cpu()   # (2, D)
        v = w[1] - w[0]
    else:                                    # MLP/MKA: use the flat parameter vector as a proxy
        v = z.flat_params().cpu()
    return v / (v.norm() + 1e-9)


def _assert_not_clones(evaluators: list[Evaluator], clone_cos_max: float) -> None:
    """Fail loudly if bagging did not actually diversify the certifiers."""
    dirs = [_discriminant(e.probes) for e in evaluators]
    max_cos = 0.0
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            max_cos = max(max_cos, float(torch.dot(dirs[i], dirs[j]).abs()))
    if len(dirs) > 1 and max_cos > clone_cos_max:
        raise EvaluatorQualityError(
            f"evaluators are near-clones (max |cos|={max_cos:.5f} > "
            f"{clone_cos_max}); bagging failed to diversify certifiers, so "
            f"averaging over them is a no-op. Increase the eval fold or reduce "
            f"bag_frac."
        )


# ---------------------------------------------------------------------------
# 2.4-2.5  Score edits with the certifiers; the decomposition ladder
# ---------------------------------------------------------------------------

def score_edit(
    edit: torch.Tensor,
    evaluators: list[Evaluator],
    X_inter: torch.Tensor,
    zc_inter: torch.Tensor,
    ze_inter: torch.Tensor,
    device: torch.device,
    floor: float = DECODABILITY_FLOOR,
) -> dict:
    """Mean over evaluators of the floored, nullable reliability of one edit.

    Returns {R, C, S, sd, n_defined}. `R` is None if every certifier nulled
    (WS3 floor). `sd` is the across-evaluator dispersion (battery-i input).
    """
    Rs, Cs, Ss = [], [], []
    for ev in evaluators:
        m = compute_intervention_metrics(
            ev.probes, X_pre=X_inter, X_post=edit,
            zc=zc_inter, ze=ze_inter, device=device, floor=floor,
        )
        if m.reliability is not None:
            Rs.append(m.reliability)
        if m.completeness is not None:
            Cs.append(m.completeness)
        if m.selectivity is not None:
            Ss.append(m.selectivity)
    if not Rs:
        return {"R": None, "C": None, "S": None, "sd": None, "n_defined": 0}
    mean_R = sum(Rs) / len(Rs)
    sd_R = (sum((x - mean_R) ** 2 for x in Rs) / len(Rs)) ** 0.5 if len(Rs) > 1 else 0.0
    return {
        "R": mean_R,
        "C": (sum(Cs) / len(Cs)) if Cs else None,
        "S": (sum(Ss) / len(Ss)) if Ss else None,
        "sd": sd_R,
        "n_defined": len(Rs),
    }


def _mean_defined(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _max_defined(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return max(vals) if vals else None


def score_edits(
    edits: dict[str, torch.Tensor],
    evaluators: list[Evaluator],
    X_inter: torch.Tensor,
    zc_inter: torch.Tensor,
    ze_inter: torch.Tensor,
    device: torch.device,
    floor: float = DECODABILITY_FLOOR,
) -> dict[str, dict]:
    """Score a dict of named edits {name: X_post}. Returns {name: score_edit(...)}."""
    return {
        name: score_edit(edit, evaluators, X_inter, zc_inter, ze_inter, device, floor)
        for name, edit in edits.items()
    }


# Canonical method groupings for the ladder (WS5_DESIGN §2.5).
V1_METHODS = ("INLP", "RLACE", "AlterRep", "FGSM", "PGD")
CONCEPT_METHODS = ("INLP", "RLACE")   # probe-independent anchors -> R_excl


def build_ladder(per_method: dict[str, dict]) -> dict:
    """Assemble the decomposition ladder from per-method R̄ values.

    per_method must contain, where available, the v1 methods and the 'cand' key.
    Rungs (WS5_DESIGN §2.5):
        R_v1_max  = max  over the 5 v1 methods   (F3 present)
        R_v1_mean = mean over the 5 v1 methods   (isolates max->mean)
        R_excl    = mean over {INLP, RLACE}       (F1/F2 severed: edit ⊥ candidate w)
        R_cand    = candidate subspace erasure    (candidate-specific; headline)
    """
    def R(name):
        d = per_method.get(name)
        return d["R"] if d else None

    v1 = [R(m) for m in V1_METHODS]
    excl = [R(m) for m in CONCEPT_METHODS]
    return {
        "R_v1_max": _max_defined(v1),
        "R_v1_mean": _mean_defined(v1),
        "R_excl": _mean_defined(excl),
        "R_cand": R("cand"),
        "R_rep": R("cand"),   # headline alias; reported jointly with R_excl (WS5_DESIGN §2.5)
    }


def repaired_reliability(
    probe,
    *,
    X_ref: torch.Tensor,
    X_inter: torch.Tensor,
    zc_inter: torch.Tensor,
    ze_inter: torch.Tensor,
    evaluators: list[Evaluator],
    device: torch.device,
    floor: float = DECODABILITY_FLOOR,
    rank: int = 1,
    d_override: torch.Tensor | None = None,
    concept_edits: dict[str, torch.Tensor] | None = None,
    extra_edits: dict[str, torch.Tensor] | None = None,
) -> dict:
    """Full R_rep record for one candidate probe (WS5_DESIGN §2.5, §7.1).

    Args:
        X_ref:        fixed reference set for direction extraction (arch-neutral).
        X_inter:      edit substrate + C/S measurement fold (must be disjoint
                      from the evaluator training fold — enforced by the caller).
        evaluators:   pre-built bagged certifiers (train_independent_evaluators).
        concept_edits: precomputed candidate-INDEPENDENT edits, e.g.
                      {"INLP": X_post, "RLACE": X_post}, computed ONCE per
                      (layer/α) and reused across candidates. -> R_excl.
        extra_edits:  precomputed v1 self-referential edits {"AlterRep","FGSM",
                      "PGD"} (from a v1-style validation probe) so the head-to-head
                      ladder can report R_v1_max/mean on the SAME certifiers.
        d_override:   inject an external erasure direction (C-a E2 control).

    Returns a flat dict: the ladder rungs + per-method detail + null accounting.
    """
    edits: dict[str, torch.Tensor] = {}
    cand = candidate_edit(probe, X_ref, X_inter, device, r=rank, d_override=d_override)
    if cand is not None:
        edits["cand"] = cand
    if concept_edits:
        edits.update(concept_edits)
    if extra_edits:
        edits.update(extra_edits)

    per_method = score_edits(edits, evaluators, X_inter, zc_inter, ze_inter, device, floor)
    ladder = build_ladder(per_method)

    return {
        **ladder,
        "degenerate_direction": cand is None,
        "R_rep_per_method": {k: v["R"] for k, v in per_method.items()},
        "R_rep_sd": {k: v["sd"] for k, v in per_method.items()},
        "R_rep_n_eval_defined": {k: v["n_defined"] for k, v in per_method.items()},
        "n_evaluators": len(evaluators),
        "eval_acc_zc_oos": [ev.acc_zc_oos for ev in evaluators],
        "eval_acc_ze_oos": [ev.acc_ze_oos for ev in evaluators],
    }


def concept_edits_once(
    X_inter: torch.Tensor,
    zc_inter: torch.Tensor,
    device: torch.device,
    inlp_iters: int = 10,
    rlace_rank: int = 1,
    rlace_steps: int = 500,
) -> dict[str, torch.Tensor]:
    """Compute the candidate-INDEPENDENT concept edits (INLP, RLACE) once per
    (layer/α); reuse across all candidates in that cell. These deflate the
    concept from the data directly and never see any candidate probe, so they
    are the F1/F2-clean anchors for R_excl (WS5_DESIGN §2.5).
    """
    return {
        "INLP": apply_inlp(X_inter, zc_inter, device=device, num_iters=inlp_iters),
        "RLACE": apply_rlace(X_inter, zc_inter, device=device,
                             rank=rlace_rank, steps=rlace_steps),
    }
