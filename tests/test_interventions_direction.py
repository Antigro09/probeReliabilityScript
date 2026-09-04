"""Regression test for the binary direction used by INLP and AlterRep."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.interventions import _binary_linear_direction
from src.probes import LinearProbe


def test_binary_direction_uses_logit_weight_difference():
    probe = LinearProbe(3)
    with torch.no_grad():
        probe.linear.weight[0] = torch.tensor([10.0, 0.0, 0.0])
        probe.linear.weight[1] = torch.tensor([10.0, 3.0, 4.0])
    direction = _binary_linear_direction(probe)
    assert torch.allclose(direction, torch.tensor([0.0, 0.6, 0.8]), atol=1e-6)
