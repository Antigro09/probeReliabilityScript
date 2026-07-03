"""
Tests for WS3: decodability floors in the intervention metrics.

The key regression is the v1 "Gemma mechanism": with the Ze validation
probe at chance on clean reps (acc_ze_pre ~ 0.5), the legacy selectivity
denominator was clamped to 1e-9, so any post-accuracy above chance clipped
to S = 1.0 — which propagated into R = 1.0 for all 900 Gemma gender
records. v2 must return null (None) for that cell, keep the legacy clipped
value alongside, and emit the unclipped ratio and both room denominators.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.metrics import metrics_from_accuracies, DECODABILITY_FLOOR


def test_normal_case_matches_legacy():
    m = metrics_from_accuracies(acc_zc_pre=0.9, acc_zc_post=0.6,
                                acc_ze_pre=0.85, acc_ze_post=0.8)
    assert m.completeness == pytest.approx(0.75)
    assert m.selectivity == pytest.approx(0.3 / 0.35)
    assert m.completeness == pytest.approx(m.completeness_clipped)
    assert m.selectivity == pytest.approx(m.selectivity_clipped)
    assert m.reliability == pytest.approx(m.reliability_clipped)
    assert m.completeness_raw == pytest.approx(0.75)
    assert m.zc_room == pytest.approx(0.4)
    assert m.ze_room == pytest.approx(0.35)


def test_gemma_mechanism_ze_at_chance_is_null_not_one():
    """acc_ze_pre at chance: v1 said S=1, R=1; v2 says null."""
    m = metrics_from_accuracies(acc_zc_pre=0.95, acc_zc_post=0.5,
                                acc_ze_pre=0.5, acc_ze_post=0.51)
    # v2 primary metrics: null on the Ze side and hence for R.
    assert m.selectivity is None
    assert m.reliability is None
    assert m.completeness is not None  # Zc side is healthy
    # Legacy values reproduce the v1 pathology, kept for comparability.
    assert m.selectivity_clipped == pytest.approx(1.0)
    assert m.reliability_clipped > 0.99
    # The degenerate denominator is visible in the record.
    assert m.selectivity_raw is None
    assert m.ze_room == pytest.approx(0.0)


def test_just_below_floor_is_null_but_raw_emitted():
    m = metrics_from_accuracies(acc_zc_pre=0.9, acc_zc_post=0.7,
                                acc_ze_pre=DECODABILITY_FLOOR - 0.01,
                                acc_ze_post=0.52)
    assert m.selectivity is None
    assert m.reliability is None
    # Raw ratio still computable and emitted for diagnosis.
    assert m.selectivity_raw == pytest.approx(0.02 / 0.04)


def test_at_floor_is_reported():
    m = metrics_from_accuracies(acc_zc_pre=DECODABILITY_FLOOR,
                                acc_zc_post=0.5,
                                acc_ze_pre=0.9, acc_ze_post=0.9)
    assert m.completeness is not None
    assert m.reliability is not None


def test_below_floor_zc_side():
    m = metrics_from_accuracies(acc_zc_pre=0.51, acc_zc_post=0.5,
                                acc_ze_pre=0.9, acc_ze_post=0.85)
    assert m.completeness is None
    assert m.reliability is None
    assert m.selectivity is not None


def test_unclipped_values_can_leave_unit_interval():
    # Intervention IMPROVES Zc accuracy: raw completeness is negative.
    m = metrics_from_accuracies(acc_zc_pre=0.8, acc_zc_post=0.9,
                                acc_ze_pre=0.9, acc_ze_post=0.95)
    assert m.completeness_raw == pytest.approx(-1.0 / 3.0)
    assert m.completeness == 0.0          # clipped into [0, 1] above floor
    # Ze post above pre: raw selectivity exceeds 1.
    assert m.selectivity_raw > 1.0
    assert m.selectivity == 1.0


def test_floor_must_exceed_chance():
    with pytest.raises(ValueError):
        metrics_from_accuracies(0.9, 0.6, 0.9, 0.8, chance=0.5, floor=0.5)


def test_as_dict_schema():
    m = metrics_from_accuracies(0.9, 0.6, 0.85, 0.8)
    d = m.as_dict()
    for key in ("C", "S", "R", "C_clipped", "S_clipped", "R_clipped",
                "C_raw", "S_raw", "acc_zc_pre", "acc_zc_post",
                "acc_ze_pre", "acc_ze_post", "zc_room", "ze_room",
                "floor", "chance"):
        assert key in d, f"missing schema key {key}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
