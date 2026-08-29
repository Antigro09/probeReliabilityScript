"""Canonical reviewer-revision damage and orientation scoring.

The archived metric implementation keeps legacy clipped values for historical
comparability.  This module is deliberately stricter: invalid or under-floor
denominators are represented explicitly, and label orientation can only be fit
on a calibration split before being applied to final-test logits.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

ScoreStatus = Literal[
    "ok", "pre_target_below_floor", "pre_control_below_floor", "invalid"
]


@dataclass(frozen=True)
class DamageScore:
    target_acc_pre: float
    target_acc_post: float
    control_acc_pre: float
    control_acc_post: float
    C: float | None
    S: float | None
    H: float | None
    status: ScoreStatus
    reason: str | None
    C_raw: float | None
    S_raw: float | None
    chance: float
    floor: float

    def as_dict(self) -> dict:
        return asdict(self)


def _clip_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _harmonic_mean(left: float, right: float) -> float:
    if left + right <= 0.0:
        return 0.0
    return 2.0 * left * right / (left + right)


def compute_damage_score(
    target_acc_pre: float,
    target_acc_post: float,
    control_acc_pre: float,
    control_acc_post: float,
    *,
    chance: float = 0.5,
    floor: float = 0.55,
) -> DamageScore:
    """Compute target damage ``C``, control preservation ``S``, and ``H``.

    A below-floor pre-edit accuracy makes only its corresponding ratio null.
    ``H`` is null whenever either component is null.  Malformed accuracies make
    the complete score invalid instead of allowing NaNs or clipped artifacts to
    enter an experiment row.
    """

    if not math.isfinite(chance) or not math.isfinite(floor):
        raise ValueError("chance and floor must be finite")
    if not 0.0 <= chance < floor <= 1.0:
        raise ValueError("floor must be in (chance, 1]")

    values = (
        float(target_acc_pre),
        float(target_acc_post),
        float(control_acc_pre),
        float(control_acc_post),
    )
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        return DamageScore(
            target_acc_pre=values[0],
            target_acc_post=values[1],
            control_acc_pre=values[2],
            control_acc_post=values[3],
            C=None,
            S=None,
            H=None,
            status="invalid",
            reason="accuracies must be finite values in [0, 1]",
            C_raw=None,
            S_raw=None,
            chance=chance,
            floor=floor,
        )

    target_room = values[0] - chance
    control_room = values[2] - chance
    C_raw = None if abs(target_room) < 1e-12 else (values[0] - values[1]) / target_room
    S_raw = None if abs(control_room) < 1e-12 else (values[3] - chance) / control_room
    target_valid = values[0] >= floor
    control_valid = values[2] >= floor
    C = _clip_unit(C_raw) if target_valid and C_raw is not None else None
    S = _clip_unit(S_raw) if control_valid and S_raw is not None else None
    H = _harmonic_mean(C, S) if C is not None and S is not None else None

    if not target_valid:
        status: ScoreStatus = "pre_target_below_floor"
        reason = (
            f"pre-edit target accuracy {values[0]:.12g} is below floor "
            f"{floor:.12g}"
        )
        if not control_valid:
            reason += (
                f"; pre-edit control accuracy {values[2]:.12g} is also below floor"
            )
    elif not control_valid:
        status = "pre_control_below_floor"
        reason = (
            f"pre-edit control accuracy {values[2]:.12g} is below floor "
            f"{floor:.12g}"
        )
    else:
        status = "ok"
        reason = None

    return DamageScore(
        target_acc_pre=values[0],
        target_acc_post=values[1],
        control_acc_pre=values[2],
        control_acc_post=values[3],
        C=C,
        S=S,
        H=H,
        status=status,
        reason=reason,
        C_raw=C_raw,
        S_raw=S_raw,
        chance=chance,
        floor=floor,
    )


# A short, discoverable alias for experiment code.
score_from_accuracies = compute_damage_score


@dataclass(frozen=True)
class CalibrationOrientation:
    sign: Literal[-1, 1]
    identity_accuracy: float
    flipped_accuracy: float
    selected_accuracy: float
    n: int

    def as_dict(self) -> dict:
        return asdict(self)


def _binary_margin(logits) -> torch.Tensor:
    values = torch.as_tensor(logits).detach().cpu()
    if not values.is_floating_point():
        values = values.float()
    if values.ndim == 1:
        margin = values
    elif values.ndim == 2 and values.shape[1] == 1:
        margin = values[:, 0]
    elif values.ndim == 2 and values.shape[1] == 2:
        margin = values[:, 1] - values[:, 0]
    else:
        raise ValueError("binary logits must have shape (N,), (N, 1), or (N, 2)")
    if not torch.isfinite(margin).all():
        raise ValueError("binary logits must be finite")
    return margin


def _binary_labels(labels, *, expected_n: int) -> torch.Tensor:
    values = torch.as_tensor(labels).detach().cpu().reshape(-1).long()
    if len(values) != expected_n:
        raise ValueError(f"logit/label length mismatch: {expected_n} != {len(values)}")
    if not torch.all((values == 0) | (values == 1)):
        raise ValueError("binary labels must contain only 0 and 1")
    if expected_n == 0:
        raise ValueError("cannot score an empty split")
    return values


def _predictions_from_margin(margin: torch.Tensor) -> torch.Tensor:
    # A zero margin follows torch.argmax's deterministic binary tie behavior:
    # class zero wins.
    return (margin > 0).long()


def fit_calibration_orientation(
    calibration_logits,
    calibration_labels,
) -> CalibrationOrientation:
    """Choose identity or complement using calibration labels only.

    No final-test argument is accepted by design.  Exact ties preserve the
    original evaluator orientation (``sign=+1``).
    """

    margin = _binary_margin(calibration_logits)
    labels = _binary_labels(calibration_labels, expected_n=len(margin))
    identity_predictions = _predictions_from_margin(margin)
    identity_accuracy = float((identity_predictions == labels).float().mean().item())
    flipped_accuracy = float(((1 - identity_predictions) == labels).float().mean().item())
    sign: Literal[-1, 1] = -1 if flipped_accuracy > identity_accuracy else 1
    return CalibrationOrientation(
        sign=sign,
        identity_accuracy=identity_accuracy,
        flipped_accuracy=flipped_accuracy,
        selected_accuracy=max(identity_accuracy, flipped_accuracy),
        n=len(labels),
    )


def apply_calibration_orientation(logits, orientation: CalibrationOrientation) -> torch.Tensor:
    """Apply a previously frozen orientation without accepting any labels."""

    if orientation.sign not in (-1, 1):
        raise ValueError(f"invalid orientation sign: {orientation.sign!r}")
    return _binary_margin(logits) * orientation.sign


def binary_metrics_from_logits(logits, labels) -> dict:
    """Return deterministic binary metrics from margins or two-class logits."""

    margin = _binary_margin(logits).to(torch.float64)
    label_tensor = _binary_labels(labels, expected_n=len(margin))
    predictions = _predictions_from_margin(margin)
    probabilities = torch.sigmoid(margin).numpy()
    y_true = label_tensor.numpy()
    y_pred = predictions.numpy()

    accuracy = float(accuracy_score(y_true, y_pred))
    balanced = float(balanced_accuracy_score(y_true, y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int).tolist()
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, probabilities))
    else:
        auc = None
    clipped_probabilities = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    loss = float(log_loss(y_true, clipped_probabilities, labels=[0, 1]))
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "confusion_matrix": matrix,
        "auc": auc,
        "log_loss": loss,
        "complement_sensitivity_accuracy": max(accuracy, 1.0 - accuracy),
        "n": len(y_true),
    }


def evaluate_frozen_orientation(
    final_logits,
    final_labels,
    orientation: CalibrationOrientation,
) -> dict:
    """Evaluate raw and calibration-frozen metrics on a final split."""

    raw = binary_metrics_from_logits(final_logits, final_labels)
    oriented_margin = apply_calibration_orientation(final_logits, orientation)
    oriented = binary_metrics_from_logits(oriented_margin, final_labels)
    report = {
        "orientation_sign": orientation.sign,
        "calibration_identity_accuracy": orientation.identity_accuracy,
        "calibration_flipped_accuracy": orientation.flipped_accuracy,
        "calibration_selected_accuracy": orientation.selected_accuracy,
        "calibration_n": orientation.n,
    }
    report.update({f"raw_{key}": value for key, value in raw.items()})
    report.update({f"oriented_{key}": value for key, value in oriented.items()})
    return report


__all__ = [
    "CalibrationOrientation",
    "DamageScore",
    "apply_calibration_orientation",
    "binary_metrics_from_logits",
    "compute_damage_score",
    "evaluate_frozen_orientation",
    "fit_calibration_orientation",
    "score_from_accuracies",
]
