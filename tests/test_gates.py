"""
Tests for WS2: the learnability gate (src/gates.py).

The gate must pass on representations where Zc and Ze are linearly
decodable, fail the floor check when a label is unrelated to the input
(the v1 gender signature: probes at chance), and report both checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.gates import learnability_gate
from src.probes import ProbeTrainConfig

DEVICE = torch.device("cpu")
# Production gate runs reuse the benchmark's probe config (50 epochs); this
# is a fast equivalent for synthetic 32-d data.
CFG = ProbeTrainConfig(epochs=60, lr=1e-2, weight_decay=0.01, batch_size=256)


def _learnable_data(n=2000, d=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    w_zc = torch.randn(d, generator=g)
    w_ze = torch.randn(d, generator=g)
    zc = (X @ w_zc > 0).long()
    ze = (X @ w_ze > 0).long()
    # Strengthen both signals so a 20-epoch linear probe finds them.
    X = X + 1.0 * zc.unsqueeze(1).float() * w_zc.unsqueeze(0) / w_zc.norm()
    X = X + 1.0 * ze.unsqueeze(1).float() * w_ze.unsqueeze(0) / w_ze.norm()
    return X, zc, ze


def test_gate_passes_on_learnable_task():
    X, zc, ze = _learnable_data()
    res = learnability_gate(X, zc, ze, DEVICE, CFG,
                            zc_floor=0.60, ze_floor=0.55, layer=3)
    assert res.passed, res.report()
    assert res.layer == 3
    d = res.as_dict()
    assert d["passed"] and len(d["checks"]) == 2


def test_gate_fails_on_unlearnable_zc():
    """Labels independent of X — the v1 gender failure signature."""
    X, _, ze = _learnable_data()
    g = torch.Generator().manual_seed(9)
    zc_random = torch.randint(0, 2, (X.shape[0],), generator=g)
    res = learnability_gate(X, zc_random, ze, DEVICE, CFG,
                            zc_floor=0.60, ze_floor=0.55)
    assert not res.passed
    zc_check = next(c for c in res.checks if c.feature == "zc")
    assert not zc_check.passed_floor
    ze_check = next(c for c in res.checks if c.feature == "ze")
    assert ze_check.passed, "healthy Ze side should still pass"
    assert "FAIL" in res.report()


def test_gate_fails_on_unlearnable_ze():
    X, zc, _ = _learnable_data()
    g = torch.Generator().manual_seed(10)
    ze_random = torch.randint(0, 2, (X.shape[0],), generator=g)
    res = learnability_gate(X, zc, ze_random, DEVICE, CFG,
                            zc_floor=0.60, ze_floor=0.55)
    assert not res.passed
    assert not next(c for c in res.checks if c.feature == "ze").passed_floor


def test_shuffled_control_sits_at_chance_on_healthy_data():
    X, zc, ze = _learnable_data()
    res = learnability_gate(X, zc, ze, DEVICE, CFG,
                            zc_floor=0.60, ze_floor=0.55)
    for c in res.checks:
        assert c.passed_shuffle, (
            f"shuffled control off chance on clean data: "
            f"{c.shuffled_accuracies} (margin {c.margin:.3f})"
        )


def test_margin_widens_for_small_eval_sets():
    X, zc, ze = _learnable_data(n=60)
    # (chance-level floors are irrelevant here; only the margin is checked)
    res = learnability_gate(X, zc, ze, DEVICE, CFG,
                            zc_floor=0.60, ze_floor=0.55)
    # 30 eval points -> binomial sd ~0.091 -> 4 sd ~0.37 > base 0.05
    assert res.checks[0].margin > 0.3


def test_too_few_examples_is_error():
    X, zc, ze = _learnable_data(n=30)
    try:
        learnability_gate(X, zc, ze, DEVICE, CFG,
                          zc_floor=0.6, ze_floor=0.55)
    except ValueError:
        return
    raise AssertionError("expected ValueError for tiny gate input")


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
