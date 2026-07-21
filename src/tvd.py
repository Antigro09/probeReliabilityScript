"""
WS7 — Total-Variation-Distance (TVD) reliability estimators.

WHY THIS EXISTS
---------------
Our accuracy-based completeness/selectivity (src/metrics.py) collapse the
oracle probe's soft output into a hard argmax before measuring effect. That
throws away calibration information: an intervention that pushes the oracle
from a confident (0.95, 0.05) down to a barely-decisive (0.51, 0.49) reads
as "no effect" on accuracy (both argmax to the same class) even though the
oracle's *distribution over the concept* moved almost all the way to chance.
Canby, Davies, Rastogi & Hockenmaier (arXiv:2408.15510) argue the honest
quantity is the DISTRIBUTIONAL shift of the oracle, measured with total
variation distance. This module implements their TVD-based completeness,
selectivity, and reliability ALONGSIDE the accuracy metrics, computed from
the oracle/validation probe's OUTPUT DISTRIBUTIONS (softmax of probe_logits),
never from accuracies.

    ┌────────────────────────────────────────────────────────────────────┐
    │ VERIFY vs arXiv:2408.15510 Eqs 1-4 BEFORE TRUSTING ANY NUMBER HERE. │
    │ Interpretation choices (a)-(c) below are OUR reading of the paper   │
    │ and are flagged for PI verification. If the paper disagrees, the    │
    │ formulas in this docstring — quoted verbatim — are what we coded,   │
    │ and they, not the paper, govern what these numbers mean.            │
    └────────────────────────────────────────────────────────────────────┘

EXACT FORMULAS IMPLEMENTED (quoted verbatim; TVD delta(P,Q)=0.5*sum|P-Q|)
------------------------------------------------------------------------
  Eq1 counterfactual completeness:
        C = 1 - delta(P_hat, P_star)
      P_hat  = oracle dist over Z on intervened reps,
      P_star = one-hot at target counterfactual class z'.
  Eq2 nullifying completeness:
        C = 1 - (k/(k-1)) * delta(P_hat, U)
      U      = uniform over k classes,
      P_hat  = oracle dist on intervened (nullified) reps.
  Eq3 selectivity:
        S = 1 - (1/m) * delta(P_hat, P)
      P      = oracle dist of NON-target Z_j on CLEAN reps,
      P_hat  = same after intervening on Z_i,
      m      = max(1 - min(P), max(P)).
  Eq4 reliability:
        R = 2 * C * S / (C + S).

Our pipeline's interventions (INLP, RLACE, AlterRep, FGSM, PGD) are
NULLIFYING — they drive the target concept Zc toward chance — so the
nullifying completeness (Eq2) is the primary analog of our accuracy-based
completeness. For binary k=2 the Eq2 prefactor k/(k-1) = 2. We ALSO compute
counterfactual completeness (Eq1) because AlterRep / FGSM / PGD additionally
push the representation *across* the boundary (counterfactually), so Eq1 is
the relevant completeness for those three; Eq2 is the relevant one for the
projection methods (INLP / RLACE). Both are always reported; the integrator
selects per method.

CRITICAL INTERPRETATION CHOICES — documented loudly, flagged for PI
-------------------------------------------------------------------
(a) DISTRIBUTION AGGREGATION. The oracle emits one softmax per example, but
    the TVD in Eqs 1-3 is between *distributions over the concept*, singular.
    Two reductions are offered:
      - mode='per_example' (DEFAULT): compute a per-example TVD against the
        (possibly per-example) reference, then average the TVDs.
      - mode='dist': average the per-example oracle softmaxes into an
        EXPECTED distribution P_hat = mean_i softmax(logits_i), then take a
        single TVD against the reference (the marginal / expected-distribution
        reading).
    WHICH ONE THE PAPER USES IS *TO BE VERIFIED*. The two agree only in
    degenerate cases; they can differ substantially when the oracle is
    heterogeneous across examples.

    !!! WHY THE DEFAULT IS 'per_example' AND NOT 'dist' !!!
    mode='dist' is DEGENERATE for the nullifying (Eq2) and counterfactual
    (Eq1) completeness on any balanced dataset. If labels are balanced and
    the oracle is CONFIDENT, the class-1 examples contribute a point mass at
    class 1 and the class-0 examples a point mass at class 0; their MEAN is
    the uniform distribution. So the marginal P_hat = mean_i softmax_i sits
    at (0.5, 0.5) EVEN ON CLEAN, UN-INTERVENED, PERFECTLY-DECODABLE reps —
    and Eq2 then reports C = 1 ("fully nullified") for a NO-OP intervention.
    This is exactly the WS3 floor pathology (spurious completeness ~1)
    re-entering through the aggregation choice, and the decodability floor
    does NOT catch it because the clean oracle IS decodable (acc_pre high).
    Verified numerically: on balanced separable data a no-op gives
    C_null('dist')=1.0 vs C_null('per_example')=0.0. per_example measures
    whether EACH example's distribution moved to uniform, which is what
    "nullified" means, so it is the only correct default for our nullifying
    pipeline. Do NOT switch the pipeline to 'dist' without first verifying
    against arXiv:2408.15510 AND confirming the marginal-cancellation
    degeneracy is what the paper intends (it almost certainly is not).

    NOTE on Eq1 under mode='dist': the counterfactual target z' is
    per-example (choice (c) below), so a single global one-hot P_star does
    not exist. Under 'dist' we therefore average the per-example
    counterfactual one-hots into P_star = mean_i onehot(z'_i) (the marginal
    of the flipped labels) and compare the marginal P_hat against it. This
    is itself an interpretation choice bundled into (a)+(c) and MUST be
    verified against the paper. Under mode='per_example' Eq1 is unambiguous:
    delta(softmax_i, onehot(z'_i)) per example, then averaged.

(b) DECODABILITY-FLOOR DEGENERACY (same failure mode as WS3, src/metrics.py).
    If the CLEAN oracle cannot decode the concept (pre-accuracy < floor),
    the oracle is already near-uniform on clean reps, so a nullifying
    intervention barely moves it and Eq2 completeness reads a spurious ~1
    ("already nullified"). We reuse the WS3 remedy EXACTLY: null (not clip)
    completeness when the clean Zc-oracle accuracy acc_zc_pre < floor, and
    null selectivity when acc_ze_pre < floor. Reliability is null whenever
    either is null. This mirrors the null-not-clip convention of
    InterventionMetrics so a degenerate cell is diagnosable, never silently
    optimistic.

(c) COUNTERFACTUAL TARGET z' FOR Eq1 = the OPPOSITE of the true zc per
    example (a counterfactual label flip). For binary that is 1 - zc. For
    k>2 "opposite" is ill-defined; we implement the binary flip and RAISE on
    k>2 for Eq1 rather than guess — flagged for the PI to specify the
    multi-class counterfactual target.

RELATION TO src/metrics.py (do NOT modify that file)
----------------------------------------------------
We import ValidationProbes and DECODABILITY_FLOOR from src.metrics and
probe_logits / probe_accuracy from src.probes. The oracle is
val_probes.zc_probe (for completeness) and val_probes.ze_probe (for
selectivity). Accuracies are recomputed here only to apply the floor (b);
the metrics themselves are purely distributional.

KEY FACT inherited from the benchmark: R / per_method are computed once per
layer from the validation+data probes, so — like the accuracy R — a TVD R
is IDENTICAL across all archs/seeds in a cell. Surfacing distinct-R counts
is a WS6 concern; this module just produces the per-cell numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .metrics import ValidationProbes, DECODABILITY_FLOOR
from .probes import probe_accuracy, probe_logits


# ---------------------------------------------------------------------------
# Experimenter-choice constants — lock in PREREGISTRATION_v2.md
# ---------------------------------------------------------------------------

# Chance accuracy used for the decodability floor gate (binary task = 0.5).
# lock in PREREGISTRATION_v2.md
DEFAULT_CHANCE = 0.5

# Aggregation mode for turning per-example oracle softmaxes into the
# distribution(s) that enter the TVD. See interpretation choice (a).
# 'per_example' = mean-of-per-example-TVDs; the default (the only NON-DEGENERATE
#                 choice for nullifying/counterfactual completeness on balanced
#                 data — see the banner in the module docstring).
# 'dist'        = mean-of-distributions (expected distribution). DEGENERATE for
#                 Eq1/Eq2 completeness on balanced confident data; use only after
#                 verifying against arXiv:2408.15510.
# lock in PREREGISTRATION_v2.md
DEFAULT_TVD_MODE = "per_example"


# ---------------------------------------------------------------------------
# Core TVD primitives (torch-only, dependency-light, unit-testable)
# ---------------------------------------------------------------------------

def tvd(p: torch.Tensor, q: torch.Tensor) -> float | torch.Tensor:
    """
    Total variation distance  delta(P, Q) = 0.5 * sum |P - Q|.

    Accepts either a pair of 1D probability vectors (k,), returning a Python
    float, or a pair of batched distributions (N, k), returning a (N,) tensor
    of per-row TVDs. The class dimension is always the LAST dimension.

    No normalization is performed: callers are responsible for passing valid
    probability vectors (e.g. the output of `softmax_dist`). For proper
    probability vectors the result lies in [0, 1].
    """
    p = torch.as_tensor(p)
    q = torch.as_tensor(q)
    diff = 0.5 * (p - q).abs().sum(dim=-1)
    if diff.dim() == 0:
        return float(diff)
    return diff


def softmax_dist(logits: torch.Tensor) -> torch.Tensor:
    """
    Oracle output distribution = softmax over the class dimension.

    logits: (N, k) or (k,). Returns probabilities of the same shape, with
    the last dimension summing to 1.
    """
    logits = torch.as_tensor(logits).float()
    return torch.softmax(logits, dim=-1)


def _mean_dist(probs: torch.Tensor) -> torch.Tensor:
    """Average a batch of per-example distributions (N, k) into one (k,)."""
    return probs.mean(dim=0)


def _uniform_like(k: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Uniform distribution U over k classes, shape (k,)."""
    return torch.full((k,), 1.0 / k, dtype=dtype, device=device)


def _onehot(labels: torch.Tensor, k: int) -> torch.Tensor:
    """(N,) integer labels -> (N, k) one-hot float distributions."""
    labels = labels.long().view(-1)
    oh = torch.zeros(labels.shape[0], k, dtype=torch.float32)
    oh.scatter_(1, labels.unsqueeze(1), 1.0)
    return oh


# ---------------------------------------------------------------------------
# Eq1 / Eq2 / Eq3 completeness & selectivity kernels
# ---------------------------------------------------------------------------

def nullifying_completeness(post_probs: torch.Tensor, mode: str = DEFAULT_TVD_MODE) -> float:
    """
    Eq2:  C = 1 - (k/(k-1)) * delta(P_hat, U),  U uniform over k classes,
    P_hat = oracle dist on intervened (nullified) reps.

    post_probs: (N, k) per-example oracle softmaxes on the INTERVENED reps.
    For binary k=2 the prefactor k/(k-1) = 2, so a post distribution driven
    exactly to uniform gives C = 1, and a confident point mass gives C = 0.

    DEGENERACY WARNING (mode='dist'): the marginal P_hat = mean_i post_i sits
    at uniform whenever labels are balanced and the oracle is confident, even
    with ZERO nullification, so mode='dist' reports C ~ 1 for a no-op. Prefer
    the default mode='per_example', which averages per-example TVD-to-uniform
    and correctly reports C ~ 0 for a no-op. See module docstring choice (a).
    """
    n, k = post_probs.shape
    if k < 2:
        raise ValueError("nullifying_completeness requires k >= 2 classes.")
    factor = k / (k - 1)
    U = _uniform_like(k, dtype=post_probs.dtype, device=post_probs.device)
    if mode == "dist":
        P_hat = _mean_dist(post_probs)
        d = tvd(P_hat, U)
    elif mode == "per_example":
        d = tvd(post_probs, U.unsqueeze(0)).mean().item()
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'dist' or 'per_example'.")
    return 1.0 - factor * float(d)


def counterfactual_completeness(
    post_probs: torch.Tensor,
    zc: torch.Tensor,
    mode: str = DEFAULT_TVD_MODE,
) -> float:
    """
    Eq1:  C = 1 - delta(P_hat, P_star),  P_star one-hot at target class z'.

    z' is the counterfactual flip of the true zc per example (choice (c)):
    for binary z' = 1 - zc. Raises for k > 2 (multi-class "opposite" is
    unspecified — flagged for the PI).

    post_probs: (N, k) oracle softmaxes on the INTERVENED reps.
    zc:         (N,) true target labels; z' = 1 - zc for binary.

    Aggregation (choice (a)):
      - 'per_example' (DEFAULT): mean_i delta(post_probs_i, onehot(1-zc_i)).
      - 'dist': compare marginal P_hat = mean_i post_probs_i against the
        marginal of the flipped one-hots P_star = mean_i onehot(1-zc_i).

    DEGENERACY WARNING (mode='dist'): as with Eq2, the marginals cancel on
    balanced data — both a perfect counterfactual flip AND a no-op leave the
    marginals at (0.5, 0.5) when labels are balanced, so mode='dist' cannot
    separate the two and reports C ~ 1 for both. Use the default
    mode='per_example'. See module docstring choice (a).
    """
    n, k = post_probs.shape
    if k != 2:
        raise ValueError(
            "counterfactual_completeness (Eq1) is only implemented for binary "
            "k=2, where z' = 1 - zc. For k>2 the counterfactual target is "
            "unspecified — PI must define it (interpretation choice (c))."
        )
    zc = zc.long().view(-1)
    z_prime = 1 - zc
    star = _onehot(z_prime, k)  # (N, k)
    if mode == "dist":
        P_hat = _mean_dist(post_probs)
        P_star = _mean_dist(star)
        return 1.0 - float(tvd(P_hat, P_star))
    elif mode == "per_example":
        return 1.0 - float(tvd(post_probs, star).mean())
    raise ValueError(f"unknown mode {mode!r}; expected 'dist' or 'per_example'.")


def selectivity_tvd(
    pre_probs: torch.Tensor,
    post_probs: torch.Tensor,
    mode: str = DEFAULT_TVD_MODE,
) -> float:
    """
    Eq3:  S = 1 - (1/m) * delta(P_hat, P),
      P     = oracle dist of NON-target Z_j on CLEAN reps  (pre_probs),
      P_hat = same after intervening on the target Z_i     (post_probs),
      m     = max(1 - min(P), max(P)).

    m normalizes by the largest achievable TVD from P (a point mass either
    at P's least- or most-probable class), so S = 1 iff the non-target
    distribution is unchanged and S = 0 iff it moved as far as possible.

    Under mode='dist' P and P_hat are the marginal (mean) distributions and
    m is computed from the marginal P. Under mode='per_example' the TVD and
    the normalizer m are computed per example (m_i from P_i) and averaged;
    m_i is floored away from zero to avoid division by zero when a clean
    per-example distribution is exactly uniform in a degenerate way.
    """
    if pre_probs.shape != post_probs.shape:
        raise ValueError("pre_probs and post_probs must have the same shape.")

    if mode == "dist":
        P = _mean_dist(pre_probs)
        P_hat = _mean_dist(post_probs)
        m = max(1.0 - float(P.min()), float(P.max()))
        # m == 0 only if P is simultaneously all-ones and all-zeros, which is
        # impossible for a probability vector with k>=2; guard anyway.
        m = max(m, 1e-12)
        return 1.0 - float(tvd(P_hat, P)) / m
    elif mode == "per_example":
        d = tvd(post_probs, pre_probs)            # (N,)
        m = torch.maximum(1.0 - pre_probs.min(dim=-1).values,
                          pre_probs.max(dim=-1).values)  # (N,)
        m = m.clamp_min(1e-12)
        return 1.0 - float((d / m).mean())
    raise ValueError(f"unknown mode {mode!r}; expected 'dist' or 'per_example'.")


def reliability(completeness: float | None, selectivity: float | None) -> float | None:
    """
    Eq4:  R = 2 * C * S / (C + S).

    Null-propagating (returns None if either input is None, mirroring the
    accuracy-based reliability). Returns 0.0 when C + S <= 0 so a degenerate
    but non-null pair does not divide by zero.
    """
    if completeness is None or selectivity is None:
        return None
    c, s = completeness, selectivity
    if c + s <= 0:
        return 0.0
    return 2.0 * c * s / (c + s)


# ---------------------------------------------------------------------------
# Aggregate record
# ---------------------------------------------------------------------------

@dataclass
class TVDMetrics:
    """
    TVD-based reliability metrics for one intervention on one cell.

    completeness_* / selectivity are None when the corresponding clean-oracle
    accuracy is below the decodability floor (interpretation choice (b)),
    matching the null-not-clip convention of InterventionMetrics.
    """
    # Eq2 nullifying completeness (primary analog of accuracy-based C).
    completeness_null: float | None
    # Eq1 counterfactual completeness (relevant for AlterRep/FGSM/PGD).
    completeness_cf: float | None
    # Eq3 selectivity (non-target Ze distribution preservation).
    selectivity: float | None
    # Eq4 reliability using each completeness variant.
    reliability_null: float | None
    reliability_cf: float | None
    # Clean-oracle accuracies used only for the floor gate (diagnostics).
    acc_zc_pre: float
    acc_ze_pre: float
    # Bookkeeping so a record is self-describing.
    floor: float
    chance: float
    mode: str
    num_classes: int

    def as_dict(self) -> dict:
        return {
            "C_null": self.completeness_null,
            "C_cf": self.completeness_cf,
            "S": self.selectivity,
            "R_null": self.reliability_null,
            "R_cf": self.reliability_cf,
            "acc_zc_pre": self.acc_zc_pre,
            "acc_ze_pre": self.acc_ze_pre,
            "floor": self.floor,
            "chance": self.chance,
            "mode": self.mode,
            "num_classes": self.num_classes,
        }


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def compute_tvd_metrics(
    val_probes: ValidationProbes,
    X_pre: torch.Tensor,
    X_post: torch.Tensor,
    zc: torch.Tensor,
    ze: torch.Tensor,
    device: torch.device,
    chance: float = DEFAULT_CHANCE,
    floor: float = DECODABILITY_FLOOR,
    mode: str = DEFAULT_TVD_MODE,
) -> TVDMetrics:
    """
    Evaluate TVD-based completeness / selectivity / reliability of an
    intervention, from the ORACLE probes' OUTPUT DISTRIBUTIONS.

    Args:
        val_probes: validation Zc and Ze probes trained on CLEAN reps.
                    zc_probe is the completeness oracle; ze_probe is the
                    selectivity (non-target) oracle.
        X_pre:  (N, D) representations BEFORE intervention (clean).
        X_post: (N, D) representations AFTER intervening on the target Zc.
        zc, ze: ground-truth target / non-target labels.
        chance: random-guess accuracy for the floor gate (0.5 binary).
        floor:  decodability floor; null completeness if acc_zc_pre < floor,
                null selectivity if acc_ze_pre < floor (choice (b)).
        mode:   'per_example' (mean-of-per-example-TVDs, DEFAULT — the only
                non-degenerate choice for nullifying/counterfactual completeness
                on balanced data) or 'dist' (mean-of-distributions). See
                interpretation choice (a).

    Completeness uses the Zc-oracle softmax on X_post (Eq1 & Eq2). Selectivity
    uses the Ze-oracle softmax on X_pre (P) vs X_post (P_hat) (Eq3): the
    non-target concept should be UNCHANGED by an intervention that targets Zc.
    """
    if floor <= chance + 1e-9:
        # Same invariant as metrics_from_accuracies: a floor at or below
        # chance cannot rule out a degenerate denominator.
        raise ValueError(
            f"Decodability floor ({floor}) must be strictly above chance "
            f"({chance})."
        )

    # --- Zc-oracle distributions on intervened reps (completeness) ---------
    zc_logits_post = probe_logits(val_probes.zc_probe, X_post, device)
    zc_post = softmax_dist(zc_logits_post)               # (N, k)
    k = zc_post.shape[1]

    # --- Ze-oracle distributions pre & post (selectivity, Eq3) ------------
    ze_logits_pre = probe_logits(val_probes.ze_probe, X_pre, device)
    ze_logits_post = probe_logits(val_probes.ze_probe, X_post, device)
    ze_pre = softmax_dist(ze_logits_pre)                 # (N, k)
    ze_post = softmax_dist(ze_logits_post)               # (N, k)

    # --- Decodability floor gate (choice (b)); accuracies on CLEAN reps ---
    acc_zc_pre = probe_accuracy(val_probes.zc_probe, X_pre, zc, device)
    acc_ze_pre = probe_accuracy(val_probes.ze_probe, X_pre, ze, device)

    zc_decodable = acc_zc_pre >= floor
    ze_decodable = acc_ze_pre >= floor

    # --- Completeness (Eq2 nullifying; Eq1 counterfactual) ----------------
    if zc_decodable:
        C_null = nullifying_completeness(zc_post, mode=mode)
        # Eq1 only defined for binary here (choice (c)); leave None otherwise.
        C_cf = (counterfactual_completeness(zc_post, zc, mode=mode)
                if k == 2 else None)
    else:
        C_null = None
        C_cf = None

    # --- Selectivity (Eq3) -------------------------------------------------
    S = selectivity_tvd(ze_pre, ze_post, mode=mode) if ze_decodable else None

    # --- Reliability (Eq4), null-propagating ------------------------------
    R_null = reliability(C_null, S)
    R_cf = reliability(C_cf, S)

    return TVDMetrics(
        completeness_null=C_null,
        completeness_cf=C_cf,
        selectivity=S,
        reliability_null=R_null,
        reliability_cf=R_cf,
        acc_zc_pre=acc_zc_pre,
        acc_ze_pre=acc_ze_pre,
        floor=floor,
        chance=chance,
        mode=mode,
        num_classes=k,
    )
