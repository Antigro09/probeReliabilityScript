"""Scientific experiment primitives for the locked reviewer revision.

The CLI composes these functions with cache reconstruction and immutable shard
writers.  The functions here deliberately operate on tensors and probes only,
which keeps the shared-edit and attack-integrity invariants directly testable.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.interventions import (
    apply_alterrep,
    apply_fgsm,
    apply_inlp,
    apply_pgd,
    apply_rlace,
)
from src.metrics import ValidationProbes, train_validation_probes
from src.probes import ProbeTrainConfig
from src.repro import set_seed
from src.ws5_repaired import candidate_direction_details

from .artifacts import sha256_tensor
from .scoring import (
    apply_calibration_orientation,
    binary_metrics_from_logits,
    compute_damage_score,
    fit_calibration_orientation,
)
from .training import evaluate_logits, state_dict_sha256

MATCHED_SPLIT_KEY_COLUMNS = (
    "model_key",
    "task",
    "layer",
    "pair_seed",
    "method",
    "condition",
)
EPSILON_KEY_COLUMNS = (
    "model_key",
    "task",
    "layer",
    "pair_seed",
    "method",
    "epsilon",
    "condition",
)


@dataclass(frozen=True)
class ProbePair:
    target_probe: torch.nn.Module
    control_probe: torch.nn.Module
    target_checkpoint_hash: str
    control_checkpoint_hash: str
    seed: int


@dataclass(frozen=True)
class AttackerEvaluatorPair:
    attacker: ProbePair
    evaluator: ProbePair
    pair_seed: int


def _model_state_hash(model: torch.nn.Module) -> str:
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    return state_dict_sha256(state)


def train_probe_pair(
    X: torch.Tensor,
    target_labels: torch.Tensor,
    control_labels: torch.Tensor,
    *,
    seed: int,
    device: torch.device,
    config: ProbeTrainConfig | None = None,
) -> ProbePair:
    """Retrain the canonical fixed-epoch target/control probe pair."""

    config = config or ProbeTrainConfig()
    set_seed(seed)
    pair: ValidationProbes = train_validation_probes(
        X, target_labels, control_labels, config, device, min_acc=0.0
    )
    return ProbePair(
        target_probe=pair.zc_probe,
        control_probe=pair.ze_probe,
        target_checkpoint_hash=_model_state_hash(pair.zc_probe),
        control_checkpoint_hash=_model_state_hash(pair.ze_probe),
        seed=seed,
    )


def train_attacker_evaluator_pair(
    X_attacker: torch.Tensor,
    target_attacker: torch.Tensor,
    control_attacker: torch.Tensor,
    X_evaluator: torch.Tensor,
    target_evaluator: torch.Tensor,
    control_evaluator: torch.Tensor,
    *,
    pair_seed: int,
    device: torch.device,
    config: ProbeTrainConfig | None = None,
) -> AttackerEvaluatorPair:
    """Train disjoint canonical probe pairs with disjoint deterministic seeds."""

    attacker = train_probe_pair(
        X_attacker,
        target_attacker,
        control_attacker,
        seed=10_000 + int(pair_seed),
        device=device,
        config=config,
    )
    evaluator = train_probe_pair(
        X_evaluator,
        target_evaluator,
        control_evaluator,
        seed=20_000 + int(pair_seed),
        device=device,
        config=config,
    )
    return AttackerEvaluatorPair(attacker, evaluator, int(pair_seed))


@torch.no_grad()
def _accuracy(
    probe: torch.nn.Module,
    X: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int = 1024,
) -> float:
    correct = 0
    probe.eval()
    for start in range(0, X.shape[0], batch_size):
        xb = X[start : start + batch_size].to(device).float()
        yb = labels[start : start + batch_size].to(device)
        correct += int((probe(xb).argmax(dim=1) == yb).sum().item())
    return correct / int(X.shape[0])


def score_conditions_for_edit(
    *,
    X_pre: torch.Tensor,
    X_post: torch.Tensor,
    target_labels: torch.Tensor,
    control_labels: torch.Tensor,
    matched_target_probe: torch.nn.Module,
    matched_control_probe: torch.nn.Module,
    split_target_probe: torch.nn.Module,
    split_control_probe: torch.nn.Module,
    device: torch.device,
    common: Mapping[str, Any] | None = None,
    floor: float = 0.55,
    chance: float = 0.5,
) -> list[dict[str, Any]]:
    """Score one detached tensor twice while preserving one edit hash."""

    if X_pre.shape != X_post.shape:
        raise ValueError("pre- and post-edit tensor shapes differ")
    if not torch.isfinite(X_post).all():
        raise ValueError("edited tensor contains NaN or infinity")
    frozen_post = X_post.detach().cpu().contiguous()
    edit_hash = sha256_tensor(frozen_post)
    rows: list[dict[str, Any]] = []
    for condition, target_probe, control_probe in (
        ("matched", matched_target_probe, matched_control_probe),
        ("split", split_target_probe, split_control_probe),
    ):
        values = {
            "target_acc_pre": _accuracy(target_probe, X_pre, target_labels, device),
            "target_acc_post": _accuracy(target_probe, frozen_post, target_labels, device),
            "control_acc_pre": _accuracy(control_probe, X_pre, control_labels, device),
            "control_acc_post": _accuracy(control_probe, frozen_post, control_labels, device),
        }
        score = compute_damage_score(**values, chance=chance, floor=floor)
        row = dict(common or {})
        row.update(
            {
                "condition": condition,
                "edit_hash": edit_hash,
                **score.as_dict(),
            }
        )
        rows.append(row)
    if rows[0]["edit_hash"] != rows[1]["edit_hash"]:
        raise AssertionError("matched and split conditions did not share an edit hash")
    return rows


def pgd_step_size(epsilon: float) -> float:
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    return float(epsilon) / 5.0


def attack_edit(
    method: str,
    X: torch.Tensor,
    labels: torch.Tensor,
    probe: torch.nn.Module,
    *,
    device: torch.device,
    epsilon: float,
    pgd_steps: int = 10,
) -> torch.Tensor:
    """Apply the canonical attack and enforce the locked no-op/bound checks."""

    method = method.lower()
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if epsilon == 0:
        return X.detach().cpu().clone().contiguous()
    if method == "fgsm":
        edited = apply_fgsm(
            X,
            labels,
            validation_probe=probe,
            device=device,
            epsilon=float(epsilon),
        )
    elif method == "pgd":
        edited = apply_pgd(
            X,
            labels,
            validation_probe=probe,
            device=device,
            epsilon=float(epsilon),
            steps=int(pgd_steps),
            alpha=pgd_step_size(float(epsilon)),
        )
    else:
        raise ValueError(f"attack method must be 'fgsm' or 'pgd', got {method!r}")
    edited = edited.detach().cpu().contiguous()
    validate_linf_edit(X, edited, epsilon)
    return edited


def realized_linf_norm(X_pre: torch.Tensor, X_post: torch.Tensor) -> float:
    if X_pre.shape != X_post.shape:
        raise ValueError("pre- and post-edit tensor shapes differ")
    if X_pre.numel() == 0:
        return 0.0
    return float((X_post.cpu().float() - X_pre.cpu().float()).abs().max().item())


def validate_linf_edit(
    X_pre: torch.Tensor,
    X_post: torch.Tensor,
    epsilon: float,
    *,
    tolerance: float = 1.0e-6,
) -> float:
    if not torch.isfinite(X_post).all():
        raise ValueError("attack produced NaN or infinity")
    realized = realized_linf_norm(X_pre, X_post)
    if realized > float(epsilon) + tolerance:
        raise ValueError(
            f"attack exceeded L-infinity ball: realized={realized}, epsilon={epsilon}"
        )
    if epsilon == 0 and not torch.allclose(
        X_pre.cpu().float(), X_post.cpu().float(), atol=tolerance, rtol=0.0
    ):
        raise ValueError("epsilon-zero attack was not a no-op")
    return realized


def alterrep_edit(
    X: torch.Tensor,
    labels: torch.Tensor,
    probe: torch.nn.Module,
    *,
    device: torch.device,
    alpha: float = 1.0,
) -> torch.Tensor:
    edited = apply_alterrep(
        X, labels, validation_probe=probe, device=device, alpha=float(alpha)
    )
    if not torch.isfinite(edited).all():
        raise ValueError("AlterRep produced NaN or infinity")
    return edited.detach().cpu().contiguous()


def reference_edit(
    method: str,
    X: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
    inlp_iterations: int = 10,
    rlace_rank: int = 1,
    rlace_steps: int = 500,
) -> torch.Tensor:
    method = method.lower()
    if method == "inlp":
        edited = apply_inlp(X, labels, device=device, num_iters=inlp_iterations)
    elif method == "rlace":
        edited = apply_rlace(
            X, labels, device=device, rank=rlace_rank, steps=rlace_steps
        )
    else:
        raise ValueError(f"unknown reference method {method!r}")
    if not torch.isfinite(edited).all():
        raise ValueError(f"{method} produced NaN or infinity")
    return edited.detach().cpu().contiguous()


def rank_one_projection_edit(
    X: torch.Tensor,
    direction: torch.Tensor,
    *,
    residual_tolerance: float = 1.0e-5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Project out one normalized direction and report residual diagnostics."""

    vector = direction.detach().cpu().float().flatten()
    if vector.numel() != X.shape[1]:
        raise ValueError("direction dimension does not match representations")
    norm = float(vector.norm().item())
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("projection direction is degenerate")
    vector = vector / norm
    X_cpu = X.detach().cpu().float()
    edited = (X_cpu - (X_cpu @ vector).unsqueeze(1) * vector.unsqueeze(0)).contiguous()
    residual = edited @ vector
    maximum_absolute_residual = float(residual.abs().max().item())
    rms_residual = float(residual.square().mean().sqrt().item())
    diagnostics = {
        "input_direction_norm": norm,
        "maximum_absolute_residual": maximum_absolute_residual,
        "rms_residual": rms_residual,
        "residual_tolerance": float(residual_tolerance),
        "residual_gate_passed": maximum_absolute_residual <= residual_tolerance,
    }
    if maximum_absolute_residual > residual_tolerance:
        raise ValueError(
            "rank-one projection residual threshold exceeded: "
            f"maximum={maximum_absolute_residual}, tolerance={residual_tolerance}"
        )
    return edited, diagnostics


def candidate_rank_one_direction(
    probe: torch.nn.Module,
    X_direction_fit: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Estimate a candidate direction only on the prespecified direction fold."""

    details = candidate_direction_details(
        probe, X_direction_fit, device, r=1
    )
    basis = details.get("Q")
    if basis is None or basis.shape[1] != 1:
        raise ValueError("candidate Jacobian direction is degenerate")
    direction = basis[:, 0].detach().cpu().float()
    direction = direction / direction.norm().clamp_min(1.0e-12)
    diagnostics = {
        key: value for key, value in details.items() if key != "Q"
    }
    linear_layer = getattr(probe, "linear", None)
    if linear_layer is not None:
        weight_direction = (
            linear_layer.weight.detach().cpu().float()[1]
            - linear_layer.weight.detach().cpu().float()[0]
        )
        weight_direction = weight_direction / weight_direction.norm().clamp_min(1.0e-12)
        cosine = float(torch.dot(direction, weight_direction).abs().item())
        diagnostics["linear_weight_cosine"] = cosine
        if cosine <= 0.999:
            raise ValueError(
                f"linear Jacobian/weight direction cosine {cosine:.6f} does not exceed 0.999"
            )
    return direction, diagnostics


def assert_disjoint_named_groups(
    named_groups: Mapping[str, Iterable[Any]],
) -> None:
    """Reject any group appearing in two scientific subsets."""

    owner: dict[Any, str] = {}
    for subset, values in named_groups.items():
        for value in values:
            previous = owner.get(value)
            if previous is not None and previous != subset:
                raise ValueError(
                    f"group overlap: {value!r} appears in {previous!r} and {subset!r}"
                )
            owner[value] = subset


def _per_example_payload(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    logits_cpu = logits.detach().cpu().float()
    probabilities = torch.softmax(logits_cpu, dim=1)
    return {
        "logits": logits_cpu.tolist(),
        "probabilities": probabilities.tolist(),
        "predictions": logits_cpu.argmax(dim=1).tolist(),
        "labels": labels.detach().cpu().long().tolist(),
    }


def evaluate_fixed_evaluator_edit(
    *,
    probe: torch.nn.Module,
    X_calibration_pre: torch.Tensor,
    X_calibration_post: torch.Tensor,
    calibration_labels: torch.Tensor,
    X_final_pre: torch.Tensor,
    X_final_post: torch.Tensor,
    final_labels: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """Fit orientation on calibration only, freeze it, then score final test."""

    calibration_pre_logits = evaluate_logits(probe, X_calibration_pre, device)
    calibration_post_logits = evaluate_logits(probe, X_calibration_post, device)
    final_pre_logits = evaluate_logits(probe, X_final_pre, device)
    final_post_logits = evaluate_logits(probe, X_final_post, device)
    orientation = fit_calibration_orientation(
        calibration_post_logits, calibration_labels
    )
    final_pre_metrics = binary_metrics_from_logits(final_pre_logits, final_labels)
    final_post_raw = binary_metrics_from_logits(final_post_logits, final_labels)
    oriented_final_margin = apply_calibration_orientation(
        final_post_logits, orientation
    )
    final_post_oriented = binary_metrics_from_logits(
        oriented_final_margin, final_labels
    )
    raw_score = compute_damage_score(
        final_pre_metrics["accuracy"],
        final_post_raw["accuracy"],
        1.0,
        1.0,
    )
    orientation_score = compute_damage_score(
        final_pre_metrics["accuracy"],
        final_post_oriented["accuracy"],
        1.0,
        1.0,
    )
    return {
        "orientation": asdict(orientation),
        "final_pre_raw": final_pre_metrics,
        "final_post_raw": final_post_raw,
        "final_post_oriented": final_post_oriented,
        "C_raw": raw_score.C,
        "C_orientation": orientation_score.C,
        "per_example": {
            "calibration_pre": _per_example_payload(
                calibration_pre_logits, calibration_labels
            ),
            "calibration_post": _per_example_payload(
                calibration_post_logits, calibration_labels
            ),
            "final_pre": _per_example_payload(final_pre_logits, final_labels),
            "final_post": _per_example_payload(final_post_logits, final_labels),
            "final_post_oriented_margin": oriented_final_margin.detach().cpu().tolist(),
        },
    }


def validate_unique_rows(
    rows: Iterable[Mapping[str, Any]],
    key_columns: tuple[str, ...] = MATCHED_SPLIT_KEY_COLUMNS,
) -> None:
    seen: set[tuple[Any, ...]] = set()
    for number, row in enumerate(rows):
        missing = [column for column in key_columns if column not in row]
        if missing:
            raise ValueError(f"row {number} is missing key columns {missing}")
        key = tuple(row[column] for column in key_columns)
        if key in seen:
            raise ValueError(f"duplicate scientific row key: {key}")
        seen.add(key)


def _baseline_cells_summary(cells: Any) -> dict[str, Any]:
    """Recompute baseline statistics from a 12-cell per-pair mapping."""

    if not isinstance(cells, dict):
        raise TypeError("baseline has no cells mapping")
    matched: list[float] = []
    split: list[float] = []
    gaps: list[float] = []
    pair_counts: list[tuple[int, int]] = []
    ceiling_pair_values = 0

    def pair_values(
        cell_key: str,
        method_name: str,
        method: Mapping[str, Any],
        condition: str,
    ) -> list[float]:
        values = method.get(f"{condition}_C_pairs")
        if not isinstance(values, list) or len(values) != 5:
            raise ValueError(
                f"archived cell {cell_key!r} {method_name} {condition} "
                "does not contain five pair values"
            )
        output = [float(value) for value in values]
        if not all(math.isfinite(value) for value in output):
            raise ValueError(f"archived cell {cell_key!r} contains non-finite pair values")
        if int(method.get(f"{condition}_n", -1)) != 5:
            raise ValueError(
                f"archived cell {cell_key!r} {method_name} {condition} count is not five"
            )
        stored_mean = method.get(f"{condition}_C_mean")
        recomputed_mean = float(np.mean(output))
        if stored_mean is None or not math.isclose(
            float(stored_mean), recomputed_mean, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(
                f"archived cell {cell_key!r} {method_name} {condition} "
                "stored mean disagrees with pair values"
            )
        return output

    for key in sorted(cells):
        method = cells[key].get("methods", {}).get("AlterRep")
        if not isinstance(method, dict):
            raise TypeError(f"archived cell {key!r} has no AlterRep record")
        matched_pairs = pair_values(key, "AlterRep", method, "matched")
        split_pairs = pair_values(key, "AlterRep", method, "split")
        m = float(np.mean(matched_pairs))
        s = float(np.mean(split_pairs))
        gap = m - s
        if not math.isclose(
            float(method.get("gap", float("nan"))), gap, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(f"archived cell {key!r} AlterRep stored gap is stale")
        matched.append(m)
        split.append(s)
        gaps.append(gap)
        pair_counts.append((len(matched_pairs), len(split_pairs)))

        for attack_name in ("FGSM", "PGD"):
            attack = cells[key].get("methods", {}).get(attack_name)
            if not isinstance(attack, dict):
                raise TypeError(f"archived cell {key!r} has no {attack_name} record")
            for condition in ("matched", "split"):
                values = pair_values(key, attack_name, attack, condition)
                ceiling_pair_values += len(values)
                if not all(math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1.0e-12) for value in values):
                    raise ValueError(
                        f"archived {attack_name} {condition} ceiling is not reproduced "
                        f"for cell {key!r}"
                    )
    if len(cells) != 12 or any(count != (5, 5) for count in pair_counts):
        raise ValueError("archived control does not contain 12 complete five-pair cells")
    return {
        "n_cells": len(cells),
        "matched": float(np.mean(matched)),
        "split": float(np.mean(split)),
        "gap": float(np.mean(gaps)),
        "positive_gap_cells": int(sum(value > 0 for value in gaps)),
        "zero_gap_cells": int(sum(value == 0 for value in gaps)),
        "negative_gap_cells": int(sum(value < 0 for value in gaps)),
        "fgsm_pgd_ceiling_reproduced": ceiling_pair_values == 240,
        "fgsm_pgd_ceiling_pair_values": ceiling_pair_values,
        "cells": {
            key: {
                "matched": matched[index],
                "split": split[index],
                "gap": gaps[index],
            }
            for index, key in enumerate(sorted(cells))
        },
    }


def archived_baseline_summary(path: str | Path) -> dict[str, Any]:
    """Recompute the locked baseline from archived per-pair arrays."""

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = _baseline_cells_summary(payload.get("cells"))
    summary["archive_sha256"] = _sha256_file(path)
    return summary


def validate_archived_baseline(
    path: str | Path,
    *,
    absolute_tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    expected = {
        "matched": 0.9829058376336399,
        "split": 0.639718191210888,
        "gap": 0.34318764642275196,
    }
    summary = archived_baseline_summary(path)
    deviations = {
        name: abs(float(summary[name]) - value) for name, value in expected.items()
    }
    if any(value > absolute_tolerance for value in deviations.values()):
        raise ValueError(f"archived aggregate does not reproduce locked values: {deviations}")
    if summary["positive_gap_cells"] != 7 or summary["zero_gap_cells"] != 5:
        raise ValueError("archived AlterRep gap sign pattern changed")
    if not summary["fgsm_pgd_ceiling_reproduced"]:
        raise ValueError("archived FGSM/PGD epsilon=0.5 ceiling was not reproduced")
    summary["absolute_deviations"] = deviations
    summary["status"] = "ok"
    return summary


def validate_retrained_baseline(
    archive_path: str | Path,
    current_payload: Mapping[str, Any],
    *,
    aggregate_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Compare a deterministic current-code rerun with the locked archive."""

    archived = validate_archived_baseline(archive_path)
    current = _baseline_cells_summary(current_payload.get("cells"))
    if set(current["cells"]) != set(archived["cells"]):
        raise ValueError("retrained baseline cell keys differ from the archive")

    deviations = {
        name: abs(float(current[name]) - float(archived[name]))
        for name in ("matched", "split", "gap")
    }
    if any(value > aggregate_tolerance for value in deviations.values()):
        raise ValueError(
            "retrained baseline exceeds aggregate tolerance: "
            f"tolerance={aggregate_tolerance}, deviations={deviations}"
        )

    def sign(value: float) -> int:
        if math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
            return 0
        return 1 if value > 0 else -1

    sign_mismatches = [
        key
        for key in sorted(archived["cells"])
        if sign(current["cells"][key]["gap"])
        != sign(archived["cells"][key]["gap"])
    ]
    if sign_mismatches:
        raise ValueError(
            f"retrained AlterRep cell-gap sign pattern changed: {sign_mismatches}"
        )
    return {
        "status": "ok",
        "comparison_mode": "deterministic_retraining_with_locked_splits",
        "aggregate_tolerance": aggregate_tolerance,
        "aggregate_absolute_deviations": deviations,
        "sign_pattern_matches": True,
        "archived": archived,
        "current": current,
    }


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
