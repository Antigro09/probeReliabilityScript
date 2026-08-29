from __future__ import annotations

import torch

from src.reviewer_revision.training import (
    DecoderSpec,
    evaluate_logits,
    select_linear_hyperparameters,
    train_deterministic_probe,
)


def _separable(seed: int = 7):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(96, 6, generator=generator)
    y = (x[:, 0] - 0.7 * x[:, 1] + 0.2 * x[:, 2] > 0).long()
    groups = torch.arange(96) // 2
    return x[:72], y[:72], groups[:72], x[72:], y[72:]


def test_deterministic_training_repeats_history_hash_and_logits():
    x_train, y_train, _, x_val, y_val = _separable()
    spec = DecoderSpec(
        architecture="linear",
        learning_rate=0.03,
        weight_decay=0.0,
        epochs=20,
        batch_size=16,
        patience=4,
    )

    first = train_deterministic_probe(
        x_train, y_train, x_val, y_val, spec=spec, seed=19,
        device=torch.device("cpu"),
    )
    second = train_deterministic_probe(
        x_train, y_train, x_val, y_val, spec=spec, seed=19,
        device=torch.device("cpu"),
    )

    assert first.checkpoint_sha256 == second.checkpoint_sha256
    assert first.best_epoch == second.best_epoch
    assert first.history == second.history
    assert torch.equal(
        evaluate_logits(first.model, x_val, torch.device("cpu")),
        evaluate_logits(second.model, x_val, torch.device("cpu")),
    )
    assert first.history[first.best_epoch]["validation_accuracy"] >= 0.75


def test_linear_tuning_is_group_disjoint_and_deterministic():
    x_train, y_train, groups, _, _ = _separable()
    kwargs = {
        "X": x_train,
        "y": y_train,
        "groups": groups,
        "learning_rates": [0.003, 0.03],
        "weight_decays": [0.0, 0.01],
        "epochs": 12,
        "batch_size": 16,
        "patience": 3,
        "seed": 23,
        "device": torch.device("cpu"),
    }

    first = select_linear_hyperparameters(**kwargs)
    second = select_linear_hyperparameters(**kwargs)

    train_groups = set(groups[first.train_indices].tolist())
    validation_groups = set(groups[first.validation_indices].tolist())
    assert train_groups.isdisjoint(validation_groups)
    assert first.selected_spec == second.selected_spec
    assert first.records == second.records
    assert torch.equal(first.train_indices, second.train_indices)
    assert torch.equal(first.validation_indices, second.validation_indices)
    assert len(first.records) == 4


def test_mlp_seed_changes_checkpoint_but_is_repeatable():
    x_train, y_train, _, x_val, y_val = _separable()
    spec = DecoderSpec(
        architecture="mlp",
        hidden_dim=16,
        learning_rate=0.01,
        weight_decay=0.01,
        epochs=8,
        batch_size=24,
        patience=2,
    )
    seed_zero = train_deterministic_probe(
        x_train, y_train, x_val, y_val, spec=spec, seed=0,
        device=torch.device("cpu"),
    )
    seed_zero_again = train_deterministic_probe(
        x_train, y_train, x_val, y_val, spec=spec, seed=0,
        device=torch.device("cpu"),
    )
    seed_one = train_deterministic_probe(
        x_train, y_train, x_val, y_val, spec=spec, seed=1,
        device=torch.device("cpu"),
    )

    assert seed_zero.checkpoint_sha256 == seed_zero_again.checkpoint_sha256
    assert seed_zero.checkpoint_sha256 != seed_one.checkpoint_sha256
