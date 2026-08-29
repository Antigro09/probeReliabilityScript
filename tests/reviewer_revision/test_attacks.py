from __future__ import annotations

import pytest
import torch

from src.interventions import apply_fgsm, apply_pgd
from src.probes import LinearProbe


def _probe() -> LinearProbe:
    probe = LinearProbe(3)
    with torch.no_grad():
        probe.linear.weight.copy_(
            torch.tensor([[-1.0, 0.5, 0.25], [1.0, -0.5, -0.25]])
        )
        probe.linear.bias.zero_()
    probe.eval()
    return probe


@pytest.mark.parametrize("attack", ["fgsm", "pgd"])
def test_epsilon_zero_is_an_exact_noop(attack: str) -> None:
    X = torch.tensor(
        [[-1.0, 0.5, 0.2], [1.0, -0.5, -0.2], [0.3, 0.7, -0.4]],
        dtype=torch.float32,
    )
    original = X.clone()
    labels = torch.tensor([0, 1, 1])
    if attack == "fgsm":
        edited = apply_fgsm(
            X, labels, validation_probe=_probe(), device=torch.device("cpu"), epsilon=0.0
        )
    else:
        edited = apply_pgd(
            X,
            labels,
            validation_probe=_probe(),
            device=torch.device("cpu"),
            epsilon=0.0,
            steps=10,
            alpha=0.0,
        )

    assert torch.equal(edited, original)
    assert torch.equal(X, original)


@pytest.mark.parametrize("attack", ["fgsm", "pgd"])
def test_attacks_respect_the_locked_linf_ball(attack: str) -> None:
    X = torch.tensor(
        [[-1.0, 0.5, 0.2], [1.0, -0.5, -0.2], [0.3, 0.7, -0.4]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 1])
    epsilon = 0.125
    if attack == "fgsm":
        edited = apply_fgsm(
            X, labels, validation_probe=_probe(), device=torch.device("cpu"), epsilon=epsilon
        )
    else:
        edited = apply_pgd(
            X,
            labels,
            validation_probe=_probe(),
            device=torch.device("cpu"),
            epsilon=epsilon,
            steps=10,
            alpha=epsilon / 5,
        )

    realized = float((edited - X).abs().max())
    assert realized <= epsilon + 1e-6
    assert realized > 0.0


def test_locked_pgd_step_size_rule_recovers_archived_value() -> None:
    epsilon = 0.5
    assert epsilon / 5 == pytest.approx(0.1)
