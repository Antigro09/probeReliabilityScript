"""Regression tests for the INLP and approximate-RLACE projection controls."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src import interventions

DEVICE = torch.device("cpu")


def _signal_data(n: int = 192, d: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7001)
    X = torch.randn(n, d, generator=generator)
    y = (X[:, 0] + 0.35 * X[:, 1] > 0).long()
    return X, y


def _initial_rlace_projection(d: int, rank: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    initial_u = torch.randn(d, rank) * 0.01
    initial_q = torch.linalg.qr(initial_u, mode="reduced")[0]
    return torch.eye(d) - initial_q @ initial_q.T


def test_rank_one_erasure_matches_dense_projection() -> None:
    generator = torch.Generator().manual_seed(11)
    X = torch.randn(73, 13, generator=generator)
    q = torch.randn(13, 1, generator=generator)
    q = torch.linalg.qr(q, mode="reduced")[0]
    expected = X @ (torch.eye(13) - q @ q.T)

    actual = interventions._apply_orthogonal_erasure(X, q)

    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_rlace_updates_the_adversarial_subspace_from_initialization() -> None:
    X, y = _signal_data()
    seed = 31415
    initial_projection = _initial_rlace_projection(X.shape[1], rank=1, seed=seed)

    torch.manual_seed(seed)
    learned_projection = interventions.rlace_projection(
        X,
        y,
        device=DEVICE,
        rank=1,
        steps=16,
        lr=0.03,
        inner_steps=3,
    )

    movement = torch.linalg.matrix_norm(learned_projection - initial_projection)
    assert float(movement) > 1.0e-3


def test_rlace_projection_depends_on_the_training_labels() -> None:
    X, y = _signal_data()
    shuffled = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(99))]

    torch.manual_seed(2718)
    signal_projection = interventions.rlace_projection(
        X, y, device=DEVICE, rank=1, steps=24, lr=0.03, inner_steps=3
    )
    torch.manual_seed(2718)
    shuffled_projection = interventions.rlace_projection(
        X, shuffled, device=DEVICE, rank=1, steps=24, lr=0.03, inner_steps=3
    )

    separation = torch.linalg.matrix_norm(signal_projection - shuffled_projection)
    assert float(separation) > 1.0e-2


def test_rlace_projection_is_finite_symmetric_and_idempotent() -> None:
    X, y = _signal_data(d=10)
    torch.manual_seed(1618)
    projection = interventions.rlace_projection(
        X, y, device=DEVICE, rank=2, steps=12, lr=0.02, inner_steps=2
    )

    assert torch.isfinite(projection).all()
    assert torch.allclose(projection, projection.T, atol=2e-5, rtol=2e-5)
    assert torch.allclose(
        projection @ projection, projection, atol=2e-5, rtol=2e-5
    )


def test_apply_rlace_matches_public_projection_without_dense_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y = _signal_data(n=64, d=6)
    seed = 1234
    torch.manual_seed(seed)
    projection = interventions.rlace_projection(
        X, y, device=DEVICE, rank=1, steps=4
    )
    expected = X @ projection

    def dense_constructor_was_called(*args, **kwargs):
        raise AssertionError("apply_rlace materialized the public D-by-D projection")

    monkeypatch.setattr(
        interventions, "rlace_projection", dense_constructor_was_called
    )
    torch.manual_seed(seed)
    actual = interventions.apply_rlace(X, y, device=DEVICE, rank=1, steps=4)

    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_apply_inlp_matches_public_projection_without_dense_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y = _signal_data(n=64, d=6)
    seed = 4321
    torch.manual_seed(seed)
    projection = interventions.inlp_projection(
        X, y, device=DEVICE, num_iters=1
    )
    expected = X @ projection

    def dense_constructor_was_called(*args, **kwargs):
        raise AssertionError("apply_inlp materialized the public D-by-D projection")

    monkeypatch.setattr(interventions, "inlp_projection", dense_constructor_was_called)
    torch.manual_seed(seed)
    actual = interventions.apply_inlp(X, y, device=DEVICE, num_iters=1)

    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)
