"""
Causal intervention metrics: completeness, selectivity, reliability.

Definitions (accuracy-based, following the spirit of Canby et al. 2024):

    Let acc_zc_pre, acc_zc_post be the validation Zc-probe accuracy on the
    representations BEFORE and AFTER intervention. Likewise for Ze.

    Completeness measures how much of the recoverable Zc signal the
    intervention removed:
        C = clip( (acc_zc_pre - acc_zc_post) / (acc_zc_pre - 0.5),  0, 1 )
    For binary tasks, 0.5 is chance. C = 1 means accuracy was driven all
    the way to chance; C = 0 means no effect.

    Selectivity measures how much of the unrelated Ze signal the
    intervention preserved:
        S = clip( (acc_ze_post - 0.5) / (acc_ze_pre - 0.5),  0, 1 )
    S = 1 means Ze accuracy is fully preserved; S = 0 means Ze was driven
    to chance (collateral damage).

    Reliability is the harmonic mean:
        R = 2 * C * S / (C + S),  with R = 0 when C + S = 0.

This differs from the KL-based formulation in the original notebook code,
which had a sign error (high KL was treated as low completeness, when it
should be the opposite). The accuracy-based formulation is unambiguous,
matches the Canby et al. presentation, and is what we report in the paper.

We deliberately use SEPARATE pre-trained validation probes for Zc and Ze
that were trained on UN-INTERVENED representations. This guards against
the intervention method itself silently providing the discriminative
signal (which would happen if we re-trained probes after each intervention).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .probes import LinearProbe, ProbeTrainConfig, train_probe, probe_accuracy


# ---------------------------------------------------------------------------
# Validation probes
# ---------------------------------------------------------------------------

@dataclass
class ValidationProbes:
    """Pair of probes trained on clean Zc and Ze labels for evaluation."""
    zc_probe: LinearProbe
    ze_probe: LinearProbe
    acc_zc: float
    acc_ze: float


def train_validation_probes(
    X: torch.Tensor,
    zc: torch.Tensor,
    ze: torch.Tensor,
    cfg: ProbeTrainConfig,
    device: torch.device,
    min_acc: float = 0.94,
) -> ValidationProbes:
    """
    Train two linear validation probes: one for Zc, one for Ze.
    These are LINEAR by design — we want a faithful, low-capacity readout
    so that intervention effects reflect representation structure, not
    probe over-fitting.

    Raises a warning (not error) if either probe fails to reach min_acc.
    """
    dim = X.shape[1]
    zc_probe = LinearProbe(dim).to(device)
    train_probe(zc_probe, X, zc, cfg, device)
    ze_probe = LinearProbe(dim).to(device)
    train_probe(ze_probe, X, ze, cfg, device)

    acc_zc = probe_accuracy(zc_probe, X, zc, device)
    acc_ze = probe_accuracy(ze_probe, X, ze, device)
    if acc_zc < min_acc:
        print(f"  ⚠ Zc validation probe accuracy {acc_zc:.3f} < {min_acc}")
    if acc_ze < min_acc:
        print(f"  ⚠ Ze validation probe accuracy {acc_ze:.3f} < {min_acc}")

    return ValidationProbes(
        zc_probe=zc_probe, ze_probe=ze_probe,
        acc_zc=acc_zc, acc_ze=acc_ze,
    )


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

@dataclass
class InterventionMetrics:
    completeness: float
    selectivity: float
    reliability: float
    acc_zc_pre: float
    acc_zc_post: float
    acc_ze_pre: float
    acc_ze_post: float

    def as_dict(self) -> dict:
        return {
            "C": self.completeness,
            "S": self.selectivity,
            "R": self.reliability,
            "acc_zc_pre": self.acc_zc_pre,
            "acc_zc_post": self.acc_zc_post,
            "acc_ze_pre": self.acc_ze_pre,
            "acc_ze_post": self.acc_ze_post,
        }


def _harmonic_mean(a: float, b: float) -> float:
    if a + b <= 0:
        return 0.0
    return 2.0 * a * b / (a + b)


def compute_intervention_metrics(
    val_probes: ValidationProbes,
    X_pre: torch.Tensor,
    X_post: torch.Tensor,
    zc: torch.Tensor,
    ze: torch.Tensor,
    device: torch.device,
    chance: float = 0.5,
) -> InterventionMetrics:
    """
    Evaluate completeness / selectivity / reliability of an intervention.

    Args:
        val_probes: validation Zc and Ze probes trained on clean reps
        X_pre:  (N, D) representations BEFORE intervention
        X_post: (N, D) representations AFTER intervention
        zc, ze: ground-truth labels
        chance: random-guess accuracy (0.5 for binary tasks)
    """
    acc_zc_pre = probe_accuracy(val_probes.zc_probe, X_pre, zc, device)
    acc_zc_post = probe_accuracy(val_probes.zc_probe, X_post, zc, device)
    acc_ze_pre = probe_accuracy(val_probes.ze_probe, X_pre, ze, device)
    acc_ze_post = probe_accuracy(val_probes.ze_probe, X_post, ze, device)

    # Completeness: fraction of removable Zc accuracy that was removed.
    zc_room = max(1e-9, acc_zc_pre - chance)
    C = max(0.0, min(1.0, (acc_zc_pre - acc_zc_post) / zc_room))

    # Selectivity: fraction of Ze accuracy above chance that was preserved.
    ze_room = max(1e-9, acc_ze_pre - chance)
    S = max(0.0, min(1.0, (acc_ze_post - chance) / ze_room))

    R = _harmonic_mean(C, S)

    return InterventionMetrics(
        completeness=C,
        selectivity=S,
        reliability=R,
        acc_zc_pre=acc_zc_pre,
        acc_zc_post=acc_zc_post,
        acc_ze_pre=acc_ze_pre,
        acc_ze_post=acc_ze_post,
    )
