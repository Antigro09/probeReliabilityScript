"""
WS5 — Planted-structure synthetic study (environment-shift generator).

See docs/WS5_DESIGN.md §6. The synthetic study is where the circularity ("ydqQ")
critique is answered *without ever computing A*: we plant a known causal axis
`v_c`, a label-uncorrelated selectivity axis `v_e`, and a spurious shortcut axis
`v_s`, then check whether R_rep (src/ws5_repaired.py) ranks probes by their
recovery of `v_c` while the v1 max-over-methods R cannot.

Why an ENVIRONMENT SHIFT and not a single-environment Gaussian (the fix that the
statistics review forced):
  * In a single environment where the shortcut `s` correlates with the label `y`,
    the Bayes-optimal discriminant is a MIXTURE β_c v_c + β_s v_s and `v_c` is not
    separately identifiable — "recovery of v_c" would be ill-defined and would
    drop with the manipulated α for reasons unrelated to probe quality.
  * With two environments (`alpha_train` high, `alpha_eval = 0.5`), `v_c` is the
    INVARIANT predictor and `v_s` is spurious. A probe that latches the shortcut
    is confident on `env_train` (=> high A) but its extracted direction points at
    `v_s`; certifiers trained on `env_eval` (shortcut decorrelated) reveal that
    erasing that direction does not remove the true concept => LOW R_rep. That
    high-A / low-R_rep cell is the dissociation that makes an A<->R_rep
    correlation non-trivial.

This module produces representation tensors directly (no sentences, no model
forward, no on-disk cache — synthetic examples lack the stable string identity
that `hash_examples` keys on). The reps match the {X, zc, ze} contract that the
rest of the pipeline consumes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .tasks import Task


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SyntheticConfig:
    """Generator knobs. Signal/α values are experimenter choices (WS5_DESIGN §8)."""
    D: int = 256                 # ambient dimension
    N: int = 8000                # examples PER environment
    family: str = "linear"       # "linear" | "nonlinear" (XOR 2-D concept)
    alpha_train: float = 0.7     # shortcut<->label agreement where the candidate trains
    alpha_eval: float = 0.5      # decorrelated where certifiers train / edits are scored
    vs_vc_corr: float = 0.0      # <v_s, v_c> (non-orthogonal shortcut condition; D6)
    # Margins are set so the INVARIANT concept v_c (and the selectivity feature
    # v_e) are STRONGLY decodable — Bayes accuracy Phi(mu/sigma): mu=2, sigma=1
    # -> ~0.977, so independent certifiers clear EVAL_MIN_ACC=0.90 with room.
    # Candidate confusion comes from the SHORTCUT at high alpha, not from a weak
    # concept. (Locked values: PREREGISTRATION_v2.md D6.)
    mu_c: float = 2.0            # causal margin
    mu_e: float = 2.0            # selectivity-attractor margin
    mu_s: float = 2.0            # shortcut margin
    sigma: float = 1.0           # isotropic noise sd
    seed: int = 0


# ---------------------------------------------------------------------------
# Planted directions
# ---------------------------------------------------------------------------

def _orthonormal(D: int, k: int, generator: torch.Generator) -> torch.Tensor:
    """(k, D) with orthonormal rows."""
    M = torch.randn(D, k, generator=generator)
    Q, _ = torch.linalg.qr(M)          # (D, k), orthonormal columns
    return Q.T.contiguous()            # (k, D)


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-12)


def bayes_direction(v_c, v_s, mu_c, mu_s, alpha) -> torch.Tensor:
    """Closed-form optimal linear discriminant for the linear generator under
    isotropic noise: proportional to the class-mean difference
        E[x|y=1] - E[x|y=0] = 2 mu_c v_c + 2 mu_s (2α-1) v_s.
    At alpha=0.5 this is exactly v_c (the invariant direction)."""
    return _unit(mu_c * v_c + mu_s * (2 * alpha - 1) * v_s)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _sample_linear(cfg, dirs, alpha, generator) -> dict:
    v_c, v_e, v_s = dirs["v_c"], dirs["v_e"], dirs["v_s"]
    N, D = cfg.N, cfg.D
    y = torch.randint(0, 2, (N,), generator=generator)
    e = torch.randint(0, 2, (N,), generator=generator)
    # shortcut agrees with y with prob alpha
    agree = (torch.rand(N, generator=generator) < alpha).long()
    s = torch.where(agree == 1, y, 1 - y)

    sy = (2 * y - 1).float().unsqueeze(1)
    se = (2 * e - 1).float().unsqueeze(1)
    ss = (2 * s - 1).float().unsqueeze(1)
    noise = torch.randn(N, D, generator=generator) * cfg.sigma
    X = (cfg.mu_c * sy * v_c + cfg.mu_e * se * v_e + cfg.mu_s * ss * v_s + noise)
    return {"X": X.float(), "zc": y.long(), "ze": e.long()}


def _sample_nonlinear(cfg, dirs, alpha, generator) -> dict:
    """XOR concept: zc = XOR(sa, sb) with the two latent bits placed at ±mu_c
    along v_a and v_b (bimodal, cleanly separated), so the concept is genuinely
    non-linear (the 4 quadrants form an XOR, not linearly separable) yet an MLP
    certifier can learn it. Ground truth is the 2-D subspace span(v_a, v_b).

    Drawing a,b ~ N(0,1) instead (continuous) puts most mass near the decision
    axes, where XOR labels are ambiguous under noise, and even an MLP then sits
    near chance — which stalls the whole non-linear family. The bimodal design
    fixes that while keeping the concept non-linear."""
    v_a, v_b, v_e, v_s = dirs["v_a"], dirs["v_b"], dirs["v_e"], dirs["v_s"]
    N, D = cfg.N, cfg.D
    sa = torch.randint(0, 2, (N,), generator=generator)
    sb = torch.randint(0, 2, (N,), generator=generator)
    y = (sa ^ sb).long()                          # XOR — not linearly separable
    e = torch.randint(0, 2, (N,), generator=generator)
    agree = (torch.rand(N, generator=generator) < alpha).long()
    s = torch.where(agree == 1, y, 1 - y)

    ca = ((2 * sa - 1).float() * cfg.mu_c).unsqueeze(1)   # ±mu_c along v_a
    cb = ((2 * sb - 1).float() * cfg.mu_c).unsqueeze(1)   # ±mu_c along v_b
    se = (2 * e - 1).float().unsqueeze(1)
    ss = (2 * s - 1).float().unsqueeze(1)
    noise = torch.randn(N, D, generator=generator) * cfg.sigma
    X = (ca * v_a + cb * v_b
         + cfg.mu_e * se * v_e + cfg.mu_s * ss * v_s + noise)
    return {"X": X.float(), "zc": y.long(), "ze": e.long()}


def make_planted_reps(cfg: SyntheticConfig) -> tuple[dict, dict, dict]:
    """Return (reps_train, reps_eval, truth).

    reps_* are {"X": (N,D) float32, "zc": (N,) long, "ze": (N,) long}.
    truth carries the planted directions and per-environment Bayes discriminants.
    Same directions across environments; only the shortcut agreement (alpha)
    differs, making v_c the invariant predictor.
    """
    g = torch.Generator().manual_seed(cfg.seed)

    if cfg.family == "linear":
        base = _orthonormal(cfg.D, 3, g)
        v_c, v_e, v_s = base[0], base[1], base[2]
        if cfg.vs_vc_corr != 0.0:
            rho = cfg.vs_vc_corr
            v_s = _unit(rho * v_c + math.sqrt(max(0.0, 1 - rho ** 2)) * v_s)
        dirs = {"v_c": v_c, "v_e": v_e, "v_s": v_s}
        truth = {
            "family": "linear",
            "v_c": v_c, "v_e": v_e, "v_s": v_s,
            "concept_subspace": v_c.unsqueeze(1),   # (D,1)
            "bayes_train": bayes_direction(v_c, v_s, cfg.mu_c, cfg.mu_s, cfg.alpha_train),
            "bayes_eval": bayes_direction(v_c, v_s, cfg.mu_c, cfg.mu_s, cfg.alpha_eval),
        }
        # env_train and env_eval drawn with independent sub-streams for reproducibility
        gt = torch.Generator().manual_seed(cfg.seed + 1)
        ge = torch.Generator().manual_seed(cfg.seed + 2)
        reps_train = _sample_linear(cfg, dirs, cfg.alpha_train, gt)
        reps_eval = _sample_linear(cfg, dirs, cfg.alpha_eval, ge)

    elif cfg.family == "nonlinear":
        base = _orthonormal(cfg.D, 4, g)
        v_a, v_b, v_e, v_s = base[0], base[1], base[2], base[3]
        dirs = {"v_a": v_a, "v_b": v_b, "v_e": v_e, "v_s": v_s}
        V = torch.stack([v_a, v_b], dim=1)   # (D, 2) orthonormal columns
        truth = {
            "family": "nonlinear",
            "v_a": v_a, "v_b": v_b, "v_e": v_e, "v_s": v_s,
            "concept_subspace": V,
            "bayes_train": None, "bayes_eval": None,   # not a single linear direction
        }
        gt = torch.Generator().manual_seed(cfg.seed + 1)
        ge = torch.Generator().manual_seed(cfg.seed + 2)
        reps_train = _sample_nonlinear(cfg, dirs, cfg.alpha_train, gt)
        reps_eval = _sample_nonlinear(cfg, dirs, cfg.alpha_eval, ge)
    else:
        raise ValueError(f"unknown family {cfg.family!r} (expected 'linear'|'nonlinear')")

    return reps_train, reps_eval, truth


# ---------------------------------------------------------------------------
# Fold splitting (env_eval -> ref / eval / inter, all disjoint)
# ---------------------------------------------------------------------------

def split_eval_folds(reps_eval: dict, ref_frac: float = 0.3,
                     eval_frac: float = 0.35, seed: int = 0) -> dict:
    """Split the eval environment into three disjoint folds:
        - ref:   fixed reference set for candidate-direction extraction
        - eval:  certifier training fold (bagged inside train_independent_evaluators)
        - inter: edit substrate + C/S measurement
    Candidate training uses env_train separately.
    """
    if ref_frac + eval_frac >= 1.0:
        raise ValueError("ref_frac + eval_frac must be < 1.0")
    n = reps_eval["X"].shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_ref = int(ref_frac * n)
    n_eval = int(eval_frac * n)
    idx_ref = perm[:n_ref]
    idx_eval = perm[n_ref:n_ref + n_eval]
    idx_inter = perm[n_ref + n_eval:]

    def take(idx):
        return {k: reps_eval[k][idx] for k in ("X", "zc", "ze")}

    return {"ref": take(idx_ref), "eval": take(idx_eval), "inter": take(idx_inter)}


# ---------------------------------------------------------------------------
# Ground-truth recovery
# ---------------------------------------------------------------------------

def recovery_score(d_c: torch.Tensor, v_c: torch.Tensor) -> float:
    """|cos(d_c, v_c)| for single directions. Accepts (D,) or (D,1)."""
    a = d_c.flatten().float()
    b = v_c.flatten().float()
    a = a / (a.norm() + 1e-12)
    b = b / (b.norm() + 1e-12)
    return float(torch.dot(a, b).abs())


def subspace_recovery(Q: torch.Tensor, V: torch.Tensor) -> float:
    """Mean cosine of the principal angles between the candidate subspace Q
    (D,r) and the planted concept subspace V (D,m). Both should have (near-)
    orthonormal columns; we re-orthonormalize defensively. Returns the mean of
    the min(r,m) singular values of QᵀV, each in [0,1]."""
    Qo, _ = torch.linalg.qr(Q.float())
    Vo, _ = torch.linalg.qr(V.float())
    s = torch.linalg.svdvals(Qo.T @ Vo)
    k = min(Qo.shape[1], Vo.shape[1])
    return float(s[:k].clamp(0, 1).mean())


def random_cosine_baseline(D: int) -> float:
    """E|cos(random unit vector, fixed unit vector)| in R^D ≈ sqrt(2/(πD)).
    The correct chance floor for recovery_score (NOT zero)."""
    return math.sqrt(2.0 / (math.pi * D))


# ---------------------------------------------------------------------------
# Sanity: 4-cell balance
# ---------------------------------------------------------------------------

def check_4cell_balance(reps: dict, tol: float = 0.15) -> dict:
    """Verify the (zc, ze) cell counts are within `tol` (relative) of balanced,
    so the balanced-4-cell split path is used, not the zc-only fallback
    (src/tasks.py). Returns the counts; raises if too skewed."""
    zc, ze = reps["zc"], reps["ze"]
    counts = {}
    n = zc.shape[0]
    for a in (0, 1):
        for b in (0, 1):
            counts[(a, b)] = int(((zc == a) & (ze == b)).sum())
    expected = n / 4.0
    worst = max(abs(c - expected) / expected for c in counts.values())
    if worst > tol:
        raise ValueError(
            f"(zc,ze) cells too imbalanced (worst rel. dev {worst:.2f} > {tol}); "
            f"counts={counts}. A skewed sample can drop to the zc-only split "
            f"fallback and change the estimator mid-study."
        )
    return counts


# ---------------------------------------------------------------------------
# Task bookkeeping (NOT registered in _TASK_REGISTRY: reps are made directly)
# ---------------------------------------------------------------------------

class SyntheticTask(Task):
    """Label bookkeeping only. Representation production is bypassed
    (make_planted_reps), so load() is intentionally unavailable."""

    name = "synthetic"
    chance_accuracy = 0.5
    zc_description = "planted causal concept (invariant across environments)"
    ze_description = "planted selectivity attractor (label-uncorrelated)"
    zc_gate_floor = 0.60
    ze_gate_floor = 0.55

    def load(self, paths, max_examples, seed):
        raise NotImplementedError(
            "SyntheticTask produces representations via "
            "ws5_synthetic.make_planted_reps(), not from files."
        )
