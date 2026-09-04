"""Tests for basis-invariant Hessian alignment and size-normalized nulls."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pyhessian")

from src.hessian import (
    hessian_eigen_residuals,
    random_subspace_null,
    subspace_alignment,
)
from src.probes import LinearProbe


def _as_eigvec(probe, flat):
    out = []
    offset = 0
    for parameter in probe.parameters():
        out.append(flat[offset:offset + parameter.numel()].reshape_as(parameter))
        offset += parameter.numel()
    return out


def test_subspace_alignment_is_invariant_to_basis_rotation():
    probe = LinearProbe(2)  # six parameters including the two biases
    with torch.no_grad():
        for parameter in probe.parameters():
            parameter.zero_()
        probe.linear.weight[0, 0] = 3.0
        probe.linear.weight[0, 1] = 4.0
    p = probe.num_params()
    e1 = torch.zeros(p); e1[0] = 1.0
    e2 = torch.zeros(p); e2[1] = 1.0
    q1 = (e1 + e2) / 2 ** 0.5
    q2 = (e1 - e2) / 2 ** 0.5
    axis_basis = [_as_eigvec(probe, e1), _as_eigvec(probe, e2)]
    rotated_basis = [_as_eigvec(probe, q1), _as_eigvec(probe, q2)]
    assert subspace_alignment(probe, axis_basis) == pytest.approx(1.0, abs=1e-6)
    assert subspace_alignment(probe, rotated_basis) == pytest.approx(1.0, abs=1e-6)


def test_random_subspace_null_uses_parameter_count_and_rank():
    small = random_subspace_null(0.5, num_params=100, rank=5, draws=5000, seed=1)
    large = random_subspace_null(0.5, num_params=1000, rank=5, draws=5000, seed=1)
    assert small["null_mean"] > large["null_mean"]
    assert small["rank"] == 5
    assert 0.0 <= small["percentile"] <= 1.0
    assert 0.0 < small["p_upper"] <= 1.0


def test_hessian_residual_check_runs_independent_hvp():
    probe = LinearProbe(2)
    generator = torch.Generator().manual_seed(9)
    X = torch.randn(24, 2, generator=generator)
    y = (X[:, 0] > 0).long()
    flat = torch.randn(probe.num_params(), generator=generator)
    flat = flat / flat.norm()
    result = hessian_eigen_residuals(
        probe, X, y, [0.1], [_as_eigvec(probe, flat)],
        torch.device("cpu"), top_n=1, batch_size=7,
    )
    assert len(result) == 1
    assert torch.isfinite(torch.tensor(result[0]["rayleigh_quotient"]))
    assert 0.0 <= result[0]["relative_residual"] <= 1.0
