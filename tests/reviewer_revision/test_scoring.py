from __future__ import annotations

import inspect
import math

import pytest
import torch


def test_damage_score_computes_clipped_c_s_and_h() -> None:
    from src.reviewer_revision.scoring import compute_damage_score

    score = compute_damage_score(
        target_acc_pre=0.90,
        target_acc_post=0.60,
        control_acc_pre=0.85,
        control_acc_post=0.80,
    )

    expected_c = 0.75
    expected_s = 0.30 / 0.35
    expected_h = 2 * expected_c * expected_s / (expected_c + expected_s)
    assert score.status == "ok"
    assert score.reason is None
    assert score.C == pytest.approx(expected_c)
    assert score.S == pytest.approx(expected_s)
    assert score.H == pytest.approx(expected_h)
    assert score.C_raw == pytest.approx(expected_c)
    assert score.S_raw == pytest.approx(expected_s)


def test_damage_score_clips_valid_ratios_to_unit_interval() -> None:
    from src.reviewer_revision.scoring import compute_damage_score

    score = compute_damage_score(0.80, 0.20, 0.80, 0.95)

    assert score.status == "ok"
    assert score.C_raw == pytest.approx(2.0)
    assert score.S_raw == pytest.approx(1.5)
    assert score.C == pytest.approx(1.0)
    assert score.S == pytest.approx(1.0)
    assert score.H == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("target_pre", "control_pre", "status", "missing"),
    [
        (0.54, 0.90, "pre_target_below_floor", {"C", "H"}),
        (0.90, 0.54, "pre_control_below_floor", {"S", "H"}),
        (0.50, 0.50, "pre_target_below_floor", {"C", "S", "H"}),
    ],
)
def test_damage_score_reports_floor_failures_without_perfect_clipping(
    target_pre: float,
    control_pre: float,
    status: str,
    missing: set[str],
) -> None:
    from src.reviewer_revision.scoring import compute_damage_score

    score = compute_damage_score(target_pre, 0.50, control_pre, 0.60)

    assert score.status == status
    assert score.reason
    for name in missing:
        assert getattr(score, name) is None
    if "C" not in missing:
        assert score.C is not None
    if "S" not in missing:
        assert score.S is not None


def test_damage_score_accepts_the_locked_floor_exactly() -> None:
    from src.reviewer_revision.scoring import compute_damage_score

    score = compute_damage_score(0.55, 0.50, 0.55, 0.55)

    assert score.status == "ok"
    assert score.C == pytest.approx(1.0)
    assert score.S == pytest.approx(1.0)
    assert score.H == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -0.01, 1.01])
def test_damage_score_marks_nonfinite_or_out_of_range_accuracies_invalid(bad: float) -> None:
    from src.reviewer_revision.scoring import compute_damage_score

    score = compute_damage_score(0.90, bad, 0.90, 0.80)

    assert score.status == "invalid"
    assert score.C is None
    assert score.S is None
    assert score.H is None
    assert score.reason


def test_damage_score_rejects_an_invalid_floor_configuration() -> None:
    from src.reviewer_revision.scoring import compute_damage_score

    with pytest.raises(ValueError, match="floor"):
        compute_damage_score(0.90, 0.60, 0.90, 0.80, chance=0.5, floor=0.5)


def test_calibration_detects_inversion_and_freezes_the_sign_for_final_test() -> None:
    from src.reviewer_revision.scoring import (
        apply_calibration_orientation,
        binary_metrics_from_logits,
        evaluate_frozen_orientation,
        fit_calibration_orientation,
    )

    calibration_logits = torch.tensor([4.0, 3.0, -3.0, -4.0])
    calibration_labels = torch.tensor([0, 0, 1, 1])
    orientation = fit_calibration_orientation(calibration_logits, calibration_labels)
    assert orientation.sign == -1
    assert orientation.identity_accuracy == pytest.approx(0.0)
    assert orientation.flipped_accuracy == pytest.approx(1.0)
    assert orientation.selected_accuracy == pytest.approx(1.0)

    final_logits = torch.tensor([5.0, -5.0, 2.0, -2.0])
    final_labels = torch.tensor([0, 1, 0, 1])
    signed = apply_calibration_orientation(final_logits, orientation)
    assert torch.equal(signed, -final_logits)
    report = evaluate_frozen_orientation(final_logits, final_labels, orientation)
    raw = binary_metrics_from_logits(final_logits, final_labels)
    assert report["orientation_sign"] == -1
    assert report["raw_accuracy"] == pytest.approx(raw["accuracy"])
    assert report["oriented_accuracy"] == pytest.approx(1.0)


def test_orientation_tie_breaks_toward_identity() -> None:
    from src.reviewer_revision.scoring import fit_calibration_orientation

    orientation = fit_calibration_orientation(
        torch.zeros(4), torch.tensor([0, 1, 0, 1])
    )

    assert orientation.identity_accuracy == pytest.approx(0.5)
    assert orientation.flipped_accuracy == pytest.approx(0.5)
    assert orientation.selected_accuracy == pytest.approx(0.5)
    assert orientation.sign == 1


def test_orientation_fit_has_no_final_label_input() -> None:
    from src.reviewer_revision.scoring import fit_calibration_orientation

    parameters = inspect.signature(fit_calibration_orientation).parameters
    assert tuple(parameters) == ("calibration_logits", "calibration_labels")


def test_two_logit_inputs_use_the_binary_logit_margin() -> None:
    from src.reviewer_revision.scoring import (
        apply_calibration_orientation,
        fit_calibration_orientation,
    )

    logits = torch.tensor([[4.0, 1.0], [1.0, 4.0]])
    labels = torch.tensor([1, 0])
    orientation = fit_calibration_orientation(logits, labels)

    assert orientation.sign == -1
    assert torch.equal(
        apply_calibration_orientation(logits, orientation),
        torch.tensor([3.0, -3.0]),
    )


def test_orientation_rejects_nonfinite_logits() -> None:
    from src.reviewer_revision.scoring import fit_calibration_orientation

    with pytest.raises(ValueError, match="finite"):
        fit_calibration_orientation(
            torch.tensor([0.0, float("nan")]), torch.tensor([0, 1])
        )
