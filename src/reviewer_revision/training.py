"""Deterministic probe and fresh-decoder training for the reviewer revision."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from src.probes import LinearProbe, MKAProbe, MLPProbe
from src.repro import set_seed


@dataclass(frozen=True)
class DecoderSpec:
    """A fully specified decoder training recipe."""

    architecture: str = "linear"
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-2
    epochs: int = 50
    batch_size: int = 256
    patience: int = 5
    hidden_dim: int = 256
    mka_lambda: float = 0.1
    knn_k: int = 10

    def __post_init__(self) -> None:
        if self.architecture not in {"linear", "mlp", "mka"}:
            raise ValueError(f"unsupported decoder architecture: {self.architecture!r}")
        if self.learning_rate <= 0 or self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("learning rate, epochs, and batch size must be positive")
        if self.weight_decay < 0 or self.patience < 0:
            raise ValueError("weight decay and patience must be non-negative")


@dataclass
class TrainingResult:
    model: torch.nn.Module
    spec: DecoderSpec
    seed: int
    history: list[dict[str, float | int]]
    best_epoch: int
    checkpoint_sha256: str
    state_dict: dict[str, torch.Tensor]

    def metadata(self) -> dict[str, Any]:
        return {
            "spec": asdict(self.spec),
            "seed": self.seed,
            "history": self.history,
            "best_epoch": self.best_epoch,
            "checkpoint_sha256": self.checkpoint_sha256,
        }


@dataclass(frozen=True)
class HyperparameterSelection:
    selected_spec: DecoderSpec
    records: list[dict[str, float | int]]
    train_indices: torch.Tensor
    validation_indices: torch.Tensor


def make_decoder(input_dim: int, spec: DecoderSpec) -> torch.nn.Module:
    if spec.architecture == "linear":
        return LinearProbe(input_dim)
    if spec.architecture == "mlp":
        return MLPProbe(input_dim, hidden_dim=spec.hidden_dim)
    return MKAProbe(
        input_dim,
        hidden_dim=spec.hidden_dim,
        mka_lambda=spec.mka_lambda,
        knn_k=spec.knn_k,
    )


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    """Hash checkpoint values without relying on container serialization metadata."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        header = {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
        digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


@torch.no_grad()
def evaluate_logits(
    model: torch.nn.Module,
    X: torch.Tensor,
    device: torch.device,
    batch_size: int = 1024,
) -> torch.Tensor:
    model.eval()
    outputs: list[torch.Tensor] = []
    for start in range(0, X.shape[0], batch_size):
        outputs.append(model(X[start : start + batch_size].to(device).float()).cpu())
    if not outputs:
        return torch.empty((0, 2), dtype=torch.float32)
    return torch.cat(outputs, dim=0)


def evaluate_accuracy(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> float:
    if X.shape[0] == 0:
        raise ValueError("cannot evaluate an empty tensor")
    logits = evaluate_logits(model, X, device)
    return float((logits.argmax(dim=1) == y.cpu()).float().mean().item())


def _evaluate_loss_accuracy(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    logits = evaluate_logits(model, X, device)
    loss = float(F.cross_entropy(logits, y.cpu()).item())
    accuracy = float((logits.argmax(dim=1) == y.cpu()).float().mean().item())
    return loss, accuracy


def train_deterministic_probe(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_validation: torch.Tensor,
    y_validation: torch.Tensor,
    *,
    spec: DecoderSpec,
    seed: int,
    device: torch.device,
) -> TrainingResult:
    """Train with deterministic minibatches and restore the best validation epoch."""

    if X_train.ndim != 2 or X_validation.ndim != 2:
        raise ValueError("decoder inputs must be rank-2 tensors")
    if X_train.shape[0] != y_train.shape[0] or X_validation.shape[0] != y_validation.shape[0]:
        raise ValueError("input and label lengths differ")
    if X_train.shape[0] == 0 or X_validation.shape[0] == 0:
        raise ValueError("training and validation sets must both be non-empty")
    if X_train.shape[1] != X_validation.shape[1]:
        raise ValueError("training and validation dimensions differ")

    set_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = make_decoder(X_train.shape[1], spec).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)

    history: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_key = (float("inf"), float("inf"))
    epochs_without_improvement = 0

    for epoch in range(spec.epochs):
        model.train()
        permutation = torch.randperm(X_train.shape[0], generator=generator)
        total_loss = 0.0
        total_examples = 0
        for start in range(0, X_train.shape[0], spec.batch_size):
            indices = permutation[start : start + spec.batch_size]
            xb = X_train[indices].to(device).float()
            yb = y_train[indices].to(device).long()
            logits, hidden = model(xb, return_hidden=True)
            loss = F.cross_entropy(logits, yb)
            if isinstance(model, MKAProbe) and spec.mka_lambda > 0:
                loss = loss + spec.mka_lambda * model.mka_loss(xb, hidden)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(indices.numel())
            total_examples += int(indices.numel())

        validation_loss, validation_accuracy = _evaluate_loss_accuracy(
            model, X_validation, y_validation, device
        )
        epoch_record: dict[str, float | int] = {
            "epoch": epoch,
            "training_loss": total_loss / total_examples,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        history.append(epoch_record)

        key = (validation_loss, -validation_accuracy)
        if key < best_key:
            best_key = key
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if spec.patience and epochs_without_improvement >= spec.patience:
                break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    checkpoint_hash = state_dict_sha256(best_state)
    return TrainingResult(
        model=model,
        spec=spec,
        seed=seed,
        history=history,
        best_epoch=best_epoch,
        checkpoint_sha256=checkpoint_hash,
        state_dict=best_state,
    )


def _group_disjoint_inner_split(
    y: torch.Tensor,
    groups: torch.Tensor,
    *,
    seed: int,
    validation_fraction: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    if y.shape[0] != groups.shape[0]:
        raise ValueError("labels and groups have different lengths")
    unique_groups = torch.unique(groups.cpu(), sorted=True)
    if unique_groups.numel() < 4:
        raise ValueError("at least four groups are required for inner validation")
    n_validation = max(1, round(unique_groups.numel() * validation_fraction))
    n_validation = min(n_validation, unique_groups.numel() - 1)
    y_cpu = y.cpu()
    groups_cpu = groups.cpu()

    for attempt in range(128):
        generator = torch.Generator().manual_seed(seed + attempt)
        order = torch.randperm(unique_groups.numel(), generator=generator)
        validation_groups = unique_groups[order[:n_validation]]
        validation_mask = torch.isin(groups_cpu, validation_groups)
        train_indices = (~validation_mask).nonzero(as_tuple=True)[0]
        validation_indices = validation_mask.nonzero(as_tuple=True)[0]
        if torch.unique(y_cpu[train_indices]).numel() == torch.unique(y_cpu).numel() and torch.unique(
            y_cpu[validation_indices]
        ).numel() == torch.unique(y_cpu).numel():
            return train_indices, validation_indices
    raise ValueError("could not create a class-covering group-disjoint validation split")


def select_linear_hyperparameters(
    *,
    X: torch.Tensor,
    y: torch.Tensor,
    groups: torch.Tensor,
    learning_rates: list[float],
    weight_decays: list[float],
    epochs: int,
    batch_size: int,
    patience: int,
    seed: int,
    device: torch.device,
) -> HyperparameterSelection:
    """Select once on unedited data; callers freeze the returned spec for every edit."""

    train_indices, validation_indices = _group_disjoint_inner_split(
        y, groups, seed=seed
    )
    records: list[dict[str, float | int]] = []
    candidates: list[tuple[tuple[float, float, float, float], DecoderSpec]] = []
    candidate_number = 0
    for learning_rate in learning_rates:
        for weight_decay in weight_decays:
            spec = DecoderSpec(
                architecture="linear",
                learning_rate=float(learning_rate),
                weight_decay=float(weight_decay),
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
            )
            result = train_deterministic_probe(
                X[train_indices],
                y[train_indices],
                X[validation_indices],
                y[validation_indices],
                spec=spec,
                seed=seed + candidate_number,
                device=device,
            )
            best = result.history[result.best_epoch]
            record: dict[str, float | int] = {
                "learning_rate": float(learning_rate),
                "weight_decay": float(weight_decay),
                "validation_accuracy": float(best["validation_accuracy"]),
                "validation_loss": float(best["validation_loss"]),
                "best_epoch": int(result.best_epoch),
                "seed": seed + candidate_number,
            }
            records.append(record)
            ranking = (
                -float(record["validation_accuracy"]),
                float(record["validation_loss"]),
                float(learning_rate),
                float(weight_decay),
            )
            candidates.append((ranking, spec))
            candidate_number += 1

    selected_spec = min(candidates, key=lambda item: item[0])[1]
    return HyperparameterSelection(
        selected_spec=selected_spec,
        records=records,
        train_indices=train_indices,
        validation_indices=validation_indices,
    )
