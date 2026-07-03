"""
Causal intervention metrics: completeness, selectivity, reliability.

Definitions (accuracy-based, following the spirit of Canby et al. 2024):

    Let acc_zc_pre, acc_zc_post be the validation Zc-probe accuracy on the
    representations BEFORE and AFTER intervention. Likewise for Ze.

    Completeness measures how much of the recoverable Zc signal the
    intervention removed:
        C = (acc_zc_pre - acc_zc_post) / (acc_zc_pre - chance)
    For binary tasks, chance = 0.5. C = 1 means accuracy was driven all
    the way to chance; C = 0 means no effect.

    Selectivity measures how much of the unrelated Ze signal the
    intervention preserved:
        S = (acc_ze_post - chance) / (acc_ze_pre - chance)
    S = 1 means Ze accuracy is fully preserved; S = 0 means Ze was driven
    to chance (collateral damage).

    Reliability is the harmonic mean:
        R = 2 * C * S / (C + S),  with R = 0 when C + S = 0.

Decodability floor (schema v2)
------------------------------
Both ratios are meaningless when their denominator is near zero: if the
validation probe cannot decode the feature from the CLEAN representations
(pre-accuracy at or near chance), then "fraction of signal removed /
preserved" divides by noise. In the v1 benchmark this produced
constant-1.0 selectivity on models whose Ze probe sat at chance
(acc_ze_pre ~ 0.5 -> denominator ~ 1e-9 -> any post-accuracy above chance
clipped to S = 1.0), which then propagated into R = 1.0 for every seed.

v2 therefore applies a decodability floor to BOTH denominators:

    - If acc_zc_pre < floor, completeness is None (null in JSON).
    - If acc_ze_pre < floor, selectivity is None.
    - Reliability is None whenever either input is None.

The legacy clipped values (v1 semantics) are always computed and emitted
alongside as *_clipped, so v1/v2 rows remain comparable. Raw, UNCLIPPED
ratios are also emitted as *_raw, together with both room denominators,
so the failure mode is diagnosable from the record itself instead of
requiring a re-run.

DECODABILITY_FLOOR below is a provisional default. The final value is an
experimenter decision and must be locked in PREREGISTRATION_v2.md before
the v2 benchmark launches.

We deliberately use SEPARATE pre-trained validation probes for Zc and Ze
that were trained on UN-INTERVENED representations. This guards against
the intervention method itself silently providing the discriminative
signal (which would happen if we re-trained probes after each intervention).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .probes import LinearProbe, ProbeTrainConfig, train_probe, probe_accuracy


# Provisional; lock in PREREGISTRATION_v2.md before launching the v2 run.
DECODABILITY_FLOOR = 0.55


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
    # Primary (v2) metrics: None when the corresponding denominator is
    # below the decodability floor.
    completeness: float | None
    selectivity: float | None
    reliability: float | None
    # Legacy (v1) values: always clipped to [0, 1], never null. Kept for
    # comparability with the v1 benchmark records.
    completeness_clipped: float
    selectivity_clipped: float
    reliability_clipped: float
    # Raw, unclipped ratios (None only if the denominator is degenerate
    # to the point that division is meaningless, |room| < 1e-9).
    completeness_raw: float | None
    selectivity_raw: float | None
    # Raw validation-probe accuracies.
    acc_zc_pre: float
    acc_zc_post: float
    acc_ze_pre: float
    acc_ze_post: float
    # Room denominators (pre-accuracy minus chance), unclamped.
    zc_room: float
    ze_room: float
    floor: float
    chance: float

    def as_dict(self) -> dict:
        return {
            "C": self.completeness,
            "S": self.selectivity,
            "R": self.reliability,
            "C_clipped": self.completeness_clipped,
            "S_clipped": self.selectivity_clipped,
            "R_clipped": self.reliability_clipped,
            "C_raw": self.completeness_raw,
            "S_raw": self.selectivity_raw,
            "acc_zc_pre": self.acc_zc_pre,
            "acc_zc_post": self.acc_zc_post,
            "acc_ze_pre": self.acc_ze_pre,
            "acc_ze_post": self.acc_ze_post,
            "zc_room": self.zc_room,
            "ze_room": self.ze_room,
            "floor": self.floor,
            "chance": self.chance,
        }


def _harmonic_mean(a: float, b: float) -> float:
    if a + b <= 0:
        return 0.0
    return 2.0 * a * b / (a + b)


def _safe_ratio(num: float, denom: float) -> float | None:
    """Unclipped ratio; None when the denominator is degenerate."""
    if abs(denom) < 1e-9:
        return None
    return num / denom


def metrics_from_accuracies(
    acc_zc_pre: float,
    acc_zc_post: float,
    acc_ze_pre: float,
    acc_ze_post: float,
    chance: float = 0.5,
    floor: float = DECODABILITY_FLOOR,
) -> InterventionMetrics:
    """
    Pure computation of C/S/R from the four validation-probe accuracies.

    Factored out of compute_intervention_metrics so the floor/clip/null
    logic is unit-testable without training probes.
    """
    if floor <= chance + 1e-9:
        raise ValueError(
            f"Decodability floor ({floor}) must be strictly above chance "
            f"({chance}); a floor at or below chance cannot rule out a "
            f"degenerate denominator."
        )

    zc_room = acc_zc_pre - chance
    ze_room = acc_ze_pre - chance

    C_raw = _safe_ratio(acc_zc_pre - acc_zc_post, zc_room)
    S_raw = _safe_ratio(acc_ze_post - chance, ze_room)

    # Legacy v1 semantics: clamp the denominator to at least 1e-9, clip the
    # ratio into [0, 1]. This is what produced the constant-1.0 pathology
    # and is kept ONLY for comparability with v1 records.
    C_clipped = max(0.0, min(1.0, (acc_zc_pre - acc_zc_post) / max(1e-9, zc_room)))
    S_clipped = max(0.0, min(1.0, (acc_ze_post - chance) / max(1e-9, ze_room)))
    R_clipped = _harmonic_mean(C_clipped, S_clipped)

    # v2 semantics: below the decodability floor the ratio is undefined.
    C = None if acc_zc_pre < floor else max(0.0, min(1.0, C_raw))
    S = None if acc_ze_pre < floor else max(0.0, min(1.0, S_raw))
    R = None if (C is None or S is None) else _harmonic_mean(C, S)

    return InterventionMetrics(
        completeness=C,
        selectivity=S,
        reliability=R,
        completeness_clipped=C_clipped,
        selectivity_clipped=S_clipped,
        reliability_clipped=R_clipped,
        completeness_raw=C_raw,
        selectivity_raw=S_raw,
        acc_zc_pre=acc_zc_pre,
        acc_zc_post=acc_zc_post,
        acc_ze_pre=acc_ze_pre,
        acc_ze_post=acc_ze_post,
        zc_room=zc_room,
        ze_room=ze_room,
        floor=floor,
        chance=chance,
    )


def compute_intervention_metrics(
    val_probes: ValidationProbes,
    X_pre: torch.Tensor,
    X_post: torch.Tensor,
    zc: torch.Tensor,
    ze: torch.Tensor,
    device: torch.device,
    chance: float = 0.5,
    floor: float = DECODABILITY_FLOOR,
) -> InterventionMetrics:
    """
    Evaluate completeness / selectivity / reliability of an intervention.

    Args:
        val_probes: validation Zc and Ze probes trained on clean reps
        X_pre:  (N, D) representations BEFORE intervention
        X_post: (N, D) representations AFTER intervention
        zc, ze: ground-truth labels
        chance: random-guess accuracy (0.5 for binary tasks)
        floor:  decodability floor applied to both pre-accuracies
    """
    acc_zc_pre = probe_accuracy(val_probes.zc_probe, X_pre, zc, device)
    acc_zc_post = probe_accuracy(val_probes.zc_probe, X_post, zc, device)
    acc_ze_pre = probe_accuracy(val_probes.ze_probe, X_pre, ze, device)
    acc_ze_post = probe_accuracy(val_probes.ze_probe, X_post, ze, device)

    return metrics_from_accuracies(
        acc_zc_pre=acc_zc_pre,
        acc_zc_post=acc_zc_post,
        acc_ze_pre=acc_ze_pre,
        acc_ze_post=acc_ze_post,
        chance=chance,
        floor=floor,
    )
