"""Post-registration robustness utilities for the schema-v3 audit.

The preregistered Phase-2 benchmark is intentionally frozen.  This module
implements the follow-up analyses requested during review without changing its
estimand or overwriting its records:

* orientation-robust fixed-decoder metrics (AUC, log loss, complemented
  accuracy) that do not mistake label inversion for erasure;
* fresh linear and nonlinear decoders trained after each edit;
* rank-matched random, distortion-matched random, and binary LEACE baselines;
* cross-fitted candidate edits and direction-stability diagnostics; and
* deterministic group-stratified five-way splits for candidate training,
  evaluator training, direction estimation, decoder retraining, and final
  testing.

Everything here operates on representation tensors.  Model extraction and
resume-safe result writing live in ``scripts/run_robustness_v3.py``.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .probes import (
    LinearProbe,
    MLPProbe,
    ProbeTrainConfig,
    probe_logits,
    train_probe,
)
from .repro import set_seed
from .tasks import Example
from .ws5_repaired import candidate_direction_details


# ---------------------------------------------------------------------------
# Prediction metrics that remain valid under label inversion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BinaryScores:
    n: int
    n_positive: int
    accuracy: float
    complemented_accuracy: float
    balanced_accuracy: float
    complemented_balanced_accuracy: float
    roc_auc: float
    oriented_roc_auc: float
    log_loss: float
    oriented_log_loss: float
    brier: float
    oriented_brier: float

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "n_positive": self.n_positive,
            "accuracy": self.accuracy,
            "complemented_accuracy": self.complemented_accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "complemented_balanced_accuracy": self.complemented_balanced_accuracy,
            "roc_auc": self.roc_auc,
            "oriented_roc_auc": self.oriented_roc_auc,
            "log_loss": self.log_loss,
            "oriented_log_loss": self.oriented_log_loss,
            "brier": self.brier,
            "oriented_brier": self.oriented_brier,
        }


def _finite_or_nan(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else float("nan")


def binary_scores_from_probabilities(
    probabilities: torch.Tensor | np.ndarray | Iterable[float],
    labels: torch.Tensor | np.ndarray | Iterable[int],
) -> BinaryScores:
    """Compute raw and orientation-corrected binary prediction metrics.

    ``oriented_*`` takes the better of the stated and complemented label
    orientation.  A perfectly inverted decoder therefore remains perfectly
    decodable instead of being counted as complete erasure.
    """
    p = np.asarray(
        probabilities.detach().cpu().numpy()
        if isinstance(probabilities, torch.Tensor)
        else probabilities,
        dtype=float,
    ).reshape(-1)
    y = np.asarray(
        labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else labels,
        dtype=int,
    ).reshape(-1)
    if p.shape != y.shape:
        raise ValueError(f"probabilities and labels differ: {p.shape} vs {y.shape}")
    if p.size == 0:
        raise ValueError("cannot score an empty prediction vector")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("binary metrics require labels in {0,1}")

    eps = np.finfo(np.float64).eps
    p = np.clip(p, eps, 1.0 - eps)
    pred = (p >= 0.5).astype(int)
    acc = float(accuracy_score(y, pred))
    bal = (
        float(balanced_accuracy_score(y, pred))
        if np.unique(y).size == 2
        else float("nan")
    )
    if np.unique(y).size == 2:
        auc = float(roc_auc_score(y, p))
    else:
        auc = float("nan")
    raw_ll = float(log_loss(y, p, labels=[0, 1]))
    inv_ll = float(log_loss(y, 1.0 - p, labels=[0, 1]))
    raw_brier = float(brier_score_loss(y, p))
    inv_brier = float(brier_score_loss(y, 1.0 - p))

    return BinaryScores(
        n=int(y.size),
        n_positive=int(y.sum()),
        accuracy=acc,
        complemented_accuracy=max(acc, 1.0 - acc),
        balanced_accuracy=_finite_or_nan(bal),
        complemented_balanced_accuracy=(
            max(bal, 1.0 - bal) if math.isfinite(bal) else float("nan")
        ),
        roc_auc=_finite_or_nan(auc),
        oriented_roc_auc=(max(auc, 1.0 - auc) if math.isfinite(auc) else float("nan")),
        log_loss=raw_ll,
        oriented_log_loss=min(raw_ll, inv_ll),
        brier=raw_brier,
        oriented_brier=min(raw_brier, inv_brier),
    )


def binary_scores_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> BinaryScores:
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"expected binary logits with shape (N,2), got {tuple(logits.shape)}")
    return binary_scores_from_probabilities(torch.softmax(logits.float(), dim=1)[:, 1], labels)


def score_probe_binary(probe, X: torch.Tensor, y: torch.Tensor, device: torch.device) -> BinaryScores:
    return binary_scores_from_logits(probe_logits(probe, X, device), y)


def _auc_skill(scores: dict | BinaryScores) -> float | None:
    auc = scores.oriented_roc_auc if isinstance(scores, BinaryScores) else scores.get("oriented_roc_auc")
    if auc is None or not math.isfinite(float(auc)):
        return None
    return max(0.0, min(1.0, 2.0 * (float(auc) - 0.5)))


def orientation_robust_cs(
    target_pre: dict | BinaryScores,
    target_post: dict | BinaryScores,
    control_pre: dict | BinaryScores,
    control_post: dict | BinaryScores,
    clean_auc_floor: float = 0.55,
) -> dict:
    """AUC-based completeness/selectivity that cannot reward inversion."""
    target_pre_auc = (
        target_pre.oriented_roc_auc if isinstance(target_pre, BinaryScores)
        else target_pre.get("oriented_roc_auc")
    )
    control_pre_auc = (
        control_pre.oriented_roc_auc if isinstance(control_pre, BinaryScores)
        else control_pre.get("oriented_roc_auc")
    )
    target_pre_skill = _auc_skill(target_pre)
    target_post_skill = _auc_skill(target_post)
    control_pre_skill = _auc_skill(control_pre)
    control_post_skill = _auc_skill(control_post)

    C = None
    if (
        target_pre_skill is not None
        and target_post_skill is not None
        and float(target_pre_auc) >= clean_auc_floor
        and target_pre_skill > 1e-12
    ):
        C = max(0.0, min(1.0, (target_pre_skill - target_post_skill) / target_pre_skill))

    S = None
    if (
        control_pre_skill is not None
        and control_post_skill is not None
        and float(control_pre_auc) >= clean_auc_floor
        and control_pre_skill > 1e-12
    ):
        S = max(0.0, min(1.0, control_post_skill / control_pre_skill))

    R = None
    if C is not None and S is not None:
        R = 0.0 if C + S <= 0 else 2.0 * C * S / (C + S)
    return {
        "C_auc_oriented": C,
        "S_auc_oriented": S,
        "R_auc_oriented": R,
        "target_skill_pre": target_pre_skill,
        "target_skill_post": target_post_skill,
        "control_skill_pre": control_pre_skill,
        "control_skill_post": control_post_skill,
        "clean_auc_floor": clean_auc_floor,
    }


def _mean_defined(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def evaluate_fixed_evaluators(
    evaluators,
    X_pre: torch.Tensor,
    X_post: torch.Tensor,
    zc: torch.Tensor,
    ze: torch.Tensor,
    device: torch.device,
    clean_auc_floor: float = 0.55,
) -> dict:
    """Evaluate an edit with every frozen evaluator and retain all components."""
    per_evaluator = []
    for ev in evaluators:
        zc_pre = score_probe_binary(ev.probes.zc_probe, X_pre, zc, device)
        zc_post = score_probe_binary(ev.probes.zc_probe, X_post, zc, device)
        ze_pre = score_probe_binary(ev.probes.ze_probe, X_pre, ze, device)
        ze_post = score_probe_binary(ev.probes.ze_probe, X_post, ze, device)
        oriented = orientation_robust_cs(
            zc_pre, zc_post, ze_pre, ze_post, clean_auc_floor=clean_auc_floor
        )
        per_evaluator.append({
            "seed": int(ev.seed),
            "target_pre": zc_pre.as_dict(),
            "target_post": zc_post.as_dict(),
            "control_pre": ze_pre.as_dict(),
            "control_post": ze_post.as_dict(),
            **oriented,
        })

    keys = ("C_auc_oriented", "S_auc_oriented", "R_auc_oriented")
    aggregate = {key: _mean_defined(row[key] for row in per_evaluator) for key in keys}
    for key in keys:
        vals = [row[key] for row in per_evaluator if row[key] is not None]
        aggregate[f"{key}_sd"] = float(np.std(vals, ddof=0)) if vals else None
        aggregate[f"{key}_n"] = len(vals)
    return {"aggregate": aggregate, "per_evaluator": per_evaluator}


def mean_evaluator_probabilities(
    evaluators,
    X_pre: torch.Tensor,
    X_post: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Return mean evaluator probabilities for optional per-example artifacts."""
    accum: dict[str, list[torch.Tensor]] = defaultdict(list)
    for ev in evaluators:
        for feature, probe in (("target", ev.probes.zc_probe), ("control", ev.probes.ze_probe)):
            accum[f"{feature}_pre"].append(
                torch.softmax(probe_logits(probe, X_pre, device).float(), dim=1)[:, 1]
            )
            accum[f"{feature}_post"].append(
                torch.softmax(probe_logits(probe, X_post, device).float(), dim=1)[:, 1]
            )
    return {key: torch.stack(vals).mean(dim=0) for key, vals in accum.items()}


# ---------------------------------------------------------------------------
# Fresh post-edit decoders
# ---------------------------------------------------------------------------


def _make_decoder(arch: str, dim: int, hidden_dim: int):
    if arch == "linear":
        return LinearProbe(dim)
    if arch == "mlp":
        return MLPProbe(dim, hidden_dim=hidden_dim)
    raise ValueError(f"decoder architecture must be linear or mlp, got {arch!r}")


def evaluate_retrained_decoders(
    X_train: torch.Tensor,
    zc_train: torch.Tensor,
    ze_train: torch.Tensor,
    X_test: torch.Tensor,
    zc_test: torch.Tensor,
    ze_test: torch.Tensor,
    train_cfg: ProbeTrainConfig,
    device: torch.device,
    hidden_dim: int,
    seeds: Iterable[int] = (710_001, 710_002, 710_003),
    architectures: Iterable[str] = ("linear", "mlp"),
) -> list[dict]:
    """Train fresh target/control decoders on edited representations."""
    rows = []
    for arch_idx, arch in enumerate(architectures):
        for seed_idx, seed in enumerate(seeds):
            for feature_idx, (feature, y_train, y_test) in enumerate((
                ("target", zc_train, zc_test),
                ("control", ze_train, ze_test),
            )):
                run_seed = int(seed) + 1000 * arch_idx + 100 * feature_idx + seed_idx
                set_seed(run_seed)
                decoder = _make_decoder(arch, X_train.shape[1], hidden_dim).to(device)
                train_probe(decoder, X_train, y_train, train_cfg, device)
                rows.append({
                    "feature": feature,
                    "arch": arch,
                    "seed": run_seed,
                    "train": score_probe_binary(decoder, X_train, y_train, device).as_dict(),
                    "test": score_probe_binary(decoder, X_test, y_test, device).as_dict(),
                })
    return rows


# ---------------------------------------------------------------------------
# Reusable representation edits and matched controls
# ---------------------------------------------------------------------------


@dataclass
class SubspaceEdit:
    Q: torch.Tensor
    scale: float = 1.0
    center: torch.Tensor | None = None

    def __post_init__(self):
        q = self.Q.detach().cpu().float()
        if q.ndim == 1:
            q = q.unsqueeze(1)
        self.Q = torch.linalg.qr(q, mode="reduced")[0]
        if self.center is not None:
            self.center = self.center.detach().cpu().float().reshape(1, -1)

    @property
    def rank(self) -> int:
        return int(self.Q.shape[1])

    def apply(self, X: torch.Tensor) -> torch.Tensor:
        base = X.detach().cpu().float()
        centered = base if self.center is None else base - self.center
        removed = (centered @ self.Q) @ self.Q.T
        return (base - float(self.scale) * removed).contiguous()

    def metadata(self) -> dict:
        return {"kind": "subspace", "rank": self.rank, "scale": float(self.scale)}


@dataclass
class MatrixEdit:
    P: torch.Tensor

    def __post_init__(self):
        self.P = self.P.detach().cpu().float()

    def apply(self, X: torch.Tensor) -> torch.Tensor:
        return (X.detach().cpu().float() @ self.P).contiguous()

    def metadata(self) -> dict:
        return {"kind": "matrix", "shape": list(self.P.shape)}


@dataclass
class LeaceBinaryEdit:
    mean: torch.Tensor
    cross_cov: torch.Tensor
    regression_direction: torch.Tensor
    denominator: float

    def apply(self, X: torch.Tensor) -> torch.Tensor:
        Xc = X.detach().cpu().float() - self.mean
        coefficient = (Xc @ self.regression_direction) / self.denominator
        return (Xc - coefficient.unsqueeze(1) * self.cross_cov.unsqueeze(0) + self.mean).contiguous()

    def metadata(self) -> dict:
        return {
            "kind": "leace_binary",
            "rank": 1,
            "denominator": float(self.denominator),
        }


def fit_leace_binary(X: torch.Tensor, y: torch.Tensor) -> LeaceBinaryEdit:
    """Fit the closed-form rank-one LEACE eraser for a binary concept.

    The returned affine map makes the edited representation's empirical
    cross-covariance with the centered label zero.  It uses the least-squares
    discriminant directly and avoids materializing a dense covariance inverse.
    """
    Xf = X.detach().cpu().float()
    yf = y.detach().cpu().float()
    if Xf.shape[0] < 4 or torch.unique(yf).numel() != 2:
        raise ValueError("LEACE needs at least four examples and both binary classes")
    mean = Xf.mean(dim=0, keepdim=True)
    Xc = Xf - mean
    yc = yf - yf.mean()
    regression = torch.linalg.lstsq(Xc, yc.unsqueeze(1)).solution[:, 0]
    cross_cov = (Xc.T @ yc) / Xc.shape[0]
    denominator = float(torch.dot(regression, cross_cov))
    if abs(denominator) <= 1e-12:
        raise ValueError("LEACE fit is degenerate: target has no linear covariance with X")
    return LeaceBinaryEdit(mean, cross_cov, regression, denominator)


def random_subspace(dim: int, rank: int, seed: int) -> torch.Tensor:
    if not 1 <= int(rank) <= int(dim):
        raise ValueError(f"require 1 <= rank <= dim, got {rank}, {dim}")
    generator = torch.Generator().manual_seed(int(seed))
    return torch.linalg.qr(torch.randn(dim, rank, generator=generator), mode="reduced")[0]


def distortion_norm(edit, X: torch.Tensor) -> float:
    return float((edit.apply(X) - X.detach().cpu().float()).norm())


def distortion_matched_random_edit(
    target_edit: SubspaceEdit,
    X_fit: torch.Tensor,
    seed: int,
) -> tuple[SubspaceEdit, dict]:
    """Random subspace control with matched Frobenius edit magnitude."""
    Q = random_subspace(X_fit.shape[1], target_edit.rank, seed)
    unit_random = SubspaceEdit(Q)
    target_norm = distortion_norm(target_edit, X_fit)
    random_norm = distortion_norm(unit_random, X_fit)
    scale = target_norm / random_norm if random_norm > 1e-12 else 0.0
    edit = SubspaceEdit(Q, scale=scale)
    return edit, {
        "target_distortion": target_norm,
        "unit_random_distortion": random_norm,
        "scale": float(scale),
        "matched_distortion": distortion_norm(edit, X_fit),
    }


def linear_probe_subspace(probe: LinearProbe) -> torch.Tensor:
    w = probe.linear.weight.detach().cpu().float()
    direction = w[1] - w[0]
    if float(direction.norm()) <= 1e-12:
        raise ValueError("linear probe has a degenerate discriminant")
    return (direction / direction.norm()).unsqueeze(1)


def subspace_similarity(Q1: torch.Tensor, Q2: torch.Tensor) -> float:
    q1 = torch.linalg.qr(Q1.detach().cpu().float(), mode="reduced")[0]
    q2 = torch.linalg.qr(Q2.detach().cpu().float(), mode="reduced")[0]
    singular = torch.linalg.svdvals(q1.T @ q2)
    return float(singular[:min(q1.shape[1], q2.shape[1])].mean())


def direction_stability(
    probe,
    X_ref: torch.Tensor,
    device: torch.device,
    rank: int,
    resamples: int = 5,
    fraction: float = 0.8,
    seed: int = 0,
) -> dict:
    """Estimate candidate-direction stability across reference resamples."""
    if resamples < 2:
        return {"resamples": int(resamples), "pairwise": [], "median": None}
    n = X_ref.shape[0]
    take = max(4, min(n, int(round(fraction * n))))
    bases = []
    details = []
    for i in range(resamples):
        generator = torch.Generator().manual_seed(seed + i)
        idx = torch.randperm(n, generator=generator)[:take]
        detail = candidate_direction_details(probe, X_ref[idx], device, r=rank)
        details.append({k: v for k, v in detail.items() if k != "Q"})
        if detail["Q"] is not None:
            bases.append(detail["Q"])
    pairwise = [
        subspace_similarity(bases[i], bases[j])
        for i in range(len(bases))
        for j in range(i + 1, len(bases))
    ]
    return {
        "resamples": int(resamples),
        "fraction": float(fraction),
        "n_reference": int(n),
        "n_valid": len(bases),
        "pairwise": pairwise,
        "median": float(np.median(pairwise)) if pairwise else None,
        "p05": float(np.percentile(pairwise, 5)) if pairwise else None,
        "minimum": min(pairwise) if pairwise else None,
        "details": details,
    }


def cross_fitted_candidate_edit(
    probe,
    X: torch.Tensor,
    device: torch.device,
    rank: int = 1,
    folds: int = 5,
    seed: int = 0,
) -> dict:
    """Estimate each edit direction without using the examples it scores."""
    n = int(X.shape[0])
    if folds < 2 or n < 2 * folds:
        raise ValueError(f"cross-fitting needs folds>=2 and n>=2*folds, got n={n}, folds={folds}")
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    fold_indices = torch.tensor_split(perm, folds)
    X_post = torch.empty_like(X.detach().cpu().float())
    records = []
    bases = []
    all_idx = torch.arange(n)
    for fold_idx, heldout in enumerate(fold_indices):
        mask = torch.ones(n, dtype=torch.bool)
        mask[heldout] = False
        train_idx = all_idx[mask]
        detail = candidate_direction_details(probe, X[train_idx], device, r=rank)
        Q = detail["Q"]
        if Q is None:
            return {"X_post": None, "folds": records, "degenerate": True}
        edit = SubspaceEdit(Q)
        X_post[heldout] = edit.apply(X[heldout])
        bases.append(Q)
        records.append({
            "fold": fold_idx,
            "n_fit": int(train_idx.numel()),
            "n_score": int(heldout.numel()),
            **{k: v for k, v in detail.items() if k != "Q"},
        })
    pairwise = [
        subspace_similarity(bases[i], bases[j])
        for i in range(len(bases))
        for j in range(i + 1, len(bases))
    ]
    return {
        "X_post": X_post,
        "folds": records,
        "degenerate": False,
        "fold_direction_similarity_median": float(np.median(pairwise)) if pairwise else None,
        "fold_direction_similarity_min": min(pairwise) if pairwise else None,
    }


# ---------------------------------------------------------------------------
# Group-aware splitting and linguistic leakage controls
# ---------------------------------------------------------------------------


_GENDER_PRONOUN_RE = re.compile(r"\b(?:she|her|hers|he|him|his)\b", flags=re.IGNORECASE)


def normalize_gender_minimal_pair(sentence: str) -> str:
    return _GENDER_PRONOUN_RE.sub("<PRONOUN>", sentence.strip().lower())


def example_group_id(example: Example, task_name: str, mode: str, row_index: int) -> str:
    if mode == "none":
        return f"row:{row_index}"
    if mode == "auto":
        mode = "minimal_pair" if task_name == "gender" else "sentence"
    if mode == "sentence":
        return "sentence:" + hashlib.sha256(example.sentence.encode("utf-8")).hexdigest()[:20]
    if mode == "minimal_pair":
        if task_name != "gender":
            raise ValueError("minimal_pair grouping is defined only for the gender task")
        normalized = normalize_gender_minimal_pair(example.sentence)
        return "pair:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
    if mode in {"occupation", "template"}:
        metadata = example.metadata or {}
        key = "occupation" if mode == "occupation" else "template_id"
        if key not in metadata:
            raise ValueError(
                f"{mode} grouping requires gender.metadata.jsonl; regenerate data with "
                "python -m scripts.generate_gender_data --download"
            )
        return f"{mode}:{metadata[key]}"
    raise ValueError(f"unknown group mode {mode!r}")


def grouped_stratified_split(
    examples: list[Example],
    fractions: dict[str, float],
    seed: int,
    task_name: str,
    group_mode: str = "auto",
) -> tuple[dict[str, list[Example]], dict]:
    """Assign whole groups to folds while approximately balancing (zc,ze)."""
    if not examples:
        raise ValueError("cannot split an empty dataset")
    names = list(fractions)
    if any(v <= 0 for v in fractions.values()) or not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-8):
        raise ValueError("split fractions must be positive and sum to one")

    grouped: dict[str, list[Example]] = defaultdict(list)
    group_by_object: dict[int, str] = {}
    for i, ex in enumerate(examples):
        group_id = example_group_id(ex, task_name, group_mode, i)
        grouped[group_id].append(ex)
        group_by_object[id(ex)] = group_id

    cells = sorted({(int(ex.zc), int(ex.ze)) for ex in examples})
    total_cell = {cell: sum((ex.zc, ex.ze) == cell for ex in examples) for cell in cells}
    targets = {
        name: {cell: fractions[name] * total_cell[cell] for cell in cells}
        for name in names
    }
    target_total = {name: fractions[name] * len(examples) for name in names}
    counts = {name: {cell: 0 for cell in cells} for name in names}
    totals = {name: 0 for name in names}
    assignment = {name: [] for name in names}

    rng = random.Random(seed)
    group_items = list(grouped.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    for group_id, rows in group_items:
        group_counts = {cell: sum((ex.zc, ex.ze) == cell for ex in rows) for cell in cells}
        scored = []
        for name in names:
            before_total = totals[name]
            after_total = totals[name] + len(rows)
            score = 0.25 * (
                ((after_total - target_total[name]) ** 2
                 - (before_total - target_total[name]) ** 2)
                / (target_total[name] + 1.0)
            )
            for cell in cells:
                before = counts[name][cell]
                after = counts[name][cell] + group_counts[cell]
                score += (
                    (after - targets[name][cell]) ** 2
                    - (before - targets[name][cell]) ** 2
                ) / (targets[name][cell] + 1.0)
            overflow_before = max(0.0, before_total - 1.15 * target_total[name])
            overflow_after = max(0.0, after_total - 1.15 * target_total[name])
            score += 10.0 * (
                overflow_after * overflow_after - overflow_before * overflow_before
            ) / (target_total[name] + 1.0)
            scored.append((score, totals[name] / max(target_total[name], 1.0), name))
        _, _, chosen = min(scored)
        assignment[chosen].extend(rows)
        totals[chosen] += len(rows)
        for cell in cells:
            counts[chosen][cell] += group_counts[cell]

    for name in names:
        rng.shuffle(assignment[name])
        if not assignment[name]:
            raise ValueError(f"group split produced an empty {name!r} fold")

    # Audit the group boundary explicitly.
    seen: dict[str, str] = {}
    for name, rows in assignment.items():
        for ex in rows:
            gid = group_by_object[id(ex)]
            previous = seen.setdefault(gid, name)
            if previous != name:
                raise AssertionError(f"group {gid} crossed {previous}/{name}")

    diagnostics = {
        "group_mode": group_mode,
        "n_examples": len(examples),
        "n_groups": len(grouped),
        "fold_sizes": {name: len(rows) for name, rows in assignment.items()},
        "cell_counts": {
            name: {f"{a}/{b}": int(counts[name][(a, b)]) for a, b in cells}
            for name in names
        },
    }
    return assignment, diagnostics


def position_variant(examples: list[Example], task_name: str, position: str) -> list[Example]:
    """Construct an explicit extraction-position control dataset."""
    if position == "last":
        return list(examples)
    if task_name != "gender" or position not in {"pre-pronoun", "post-pronoun"}:
        raise ValueError("only gender supports pre-pronoun/post-pronoun position controls")

    out = []
    for ex in examples:
        match = _GENDER_PRONOUN_RE.search(ex.sentence)
        if match is None:
            raise ValueError(f"gender sentence has no pronoun: {ex.sentence!r}")
        if position == "pre-pronoun":
            text = ex.sentence[:match.start()].rstrip(" ,;:-")
        else:
            text = ex.sentence[:match.end()].rstrip(" ,;:-")
        if not text:
            raise ValueError(f"position control produced an empty sentence: {ex.sentence!r}")
        out.append(Example(
            sentence=text,
            zc=ex.zc,
            ze=ex.ze,
            group=ex.group,
            metadata=ex.metadata,
        ))
    return out
