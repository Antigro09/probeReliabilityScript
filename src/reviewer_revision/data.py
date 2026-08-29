"""Deterministic data reconstruction and representation-cache validation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from src.repro import hash_examples
from src.robustness import example_group_id, grouped_stratified_split
from src.tasks import Example, get_task

PHASE2_FRACTIONS = {
    "candidate": 0.40,
    "evaluator": 0.20,
    "intervention": 0.30,
    "test": 0.10,
}

CONSTRUCT_SUBDIVISION_FRACTIONS = {
    "direction_fit": 1 / 3,
    "fresh_decoder_fit": 1 / 3,
    "orientation_calibration": 1 / 6,
    "final_test": 1 / 6,
}


class CacheSelectionError(RuntimeError):
    """No unique cache has the requested, verified provenance."""


class CacheValidationError(RuntimeError):
    """A selected cache is malformed or scientifically unusable."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_hash(example: Example) -> str:
    digest = hashlib.sha256()
    digest.update(example.sentence.encode("utf-8"))
    digest.update(bytes([int(example.zc), int(example.ze)]))
    return digest.hexdigest()


def _identity_maps(
    examples: Sequence[Example], task_name: str
) -> tuple[dict[int, str], dict[int, str], dict[int, str], dict[int, int]]:
    example_ids: dict[int, str] = {}
    group_ids: dict[int, str] = {}
    content_hashes: dict[int, str] = {}
    source_indices: dict[int, int] = {}
    for row_index, example in enumerate(examples):
        content_hash = _content_hash(example)
        # Include the stable reconstructed row index so repeated examples remain
        # distinct rows while their shared content/group hashes remain visible.
        example_id = hashlib.sha256(
            f"{row_index}\0{content_hash}".encode("ascii")
        ).hexdigest()
        example_ids[id(example)] = example_id
        group_ids[id(example)] = example_group_id(
            example, task_name=task_name, mode="auto", row_index=row_index
        )
        content_hashes[id(example)] = content_hash
        source_indices[id(example)] = row_index
    return example_ids, group_ids, content_hashes, source_indices


def _membership_hash(example_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for example_id in sorted(example_ids):
        digest.update(example_id.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class Phase2Reconstruction:
    task_name: str
    seed: int
    n_loaded: int
    n_deduplicated: int
    n_assigned: int
    n_excluded: int
    all_data_hash: str
    folds: dict[str, list[Example]]
    fold_sizes: dict[str, int]
    fold_hashes: dict[str, str]
    manifest: list[dict]


def reconstruct_phase2_folds(
    task_name: str,
    data_paths: Iterable[Path | str],
    *,
    max_examples: int | None,
    seed: int,
) -> Phase2Reconstruction:
    """Rebuild the archived Phase-2 deduplication and four-way folds exactly."""

    task = get_task(task_name)
    loaded = task.load(
        [Path(path) for path in data_paths], max_examples=max_examples, seed=seed
    )
    seen_sentences: set[str] = set()
    deduplicated: list[Example] = []
    for example in loaded:
        if example.sentence in seen_sentences:
            continue
        seen_sentences.add(example.sentence)
        deduplicated.append(example)

    # Import the archived splitter instead of maintaining a subtly divergent
    # copy.  The regression hashes below lock its output for the revision.
    from scripts.run_benchmark_v2 import split_4way

    split_values = split_4way(deduplicated, PHASE2_FRACTIONS, seed=seed)
    names = tuple(PHASE2_FRACTIONS)
    folds = {name: rows for name, rows in zip(names, split_values, strict=True)}
    example_ids, group_ids, content_hashes, source_indices = _identity_maps(
        deduplicated, task_name
    )
    all_data_hash = hash_examples(deduplicated)
    fold_hashes = {name: hash_examples(rows) for name, rows in folds.items()}
    membership_by_object: dict[int, tuple[str, int]] = {}
    for subset, rows in folds.items():
        for position, example in enumerate(rows):
            membership_by_object[id(example)] = (subset, position)
    manifest: list[dict] = []
    for example in deduplicated:
        membership = membership_by_object.get(id(example))
        included = membership is not None
        subset, position = membership if membership is not None else (None, None)
        manifest.append(
            {
                "subset": subset,
                "position_in_subset": position,
                "source_index": source_indices[id(example)],
                "example_id": example_ids[id(example)],
                "group_id": group_ids[id(example)],
                "content_hash": content_hashes[id(example)],
                "zc": int(example.zc),
                "ze": int(example.ze),
                "source_data_hash": all_data_hash,
                "subset_data_hash": fold_hashes[subset] if subset is not None else None,
                "split_seed": seed,
                "included": included,
                "exclusion_reason": (
                    None if included else "archived_phase2_four_cell_balance"
                ),
            }
        )

    n_assigned = len(membership_by_object)
    n_excluded = len(deduplicated) - n_assigned

    return Phase2Reconstruction(
        task_name=task_name,
        seed=seed,
        n_loaded=len(loaded),
        n_deduplicated=len(deduplicated),
        n_assigned=n_assigned,
        n_excluded=n_excluded,
        all_data_hash=all_data_hash,
        folds=folds,
        fold_sizes={name: len(rows) for name, rows in folds.items()},
        fold_hashes=fold_hashes,
        manifest=manifest,
    )


@dataclass(frozen=True)
class GroupedSubdivision:
    task_name: str
    seed: int
    source_data_hash: str
    folds: dict[str, list[Example]]
    manifest: list[dict]
    subset_hashes: dict[str, str]
    diagnostics: dict


def subdivide_grouped(
    examples: Sequence[Example],
    fractions: Mapping[str, float],
    *,
    seed: int,
    task_name: str,
) -> GroupedSubdivision:
    """Split whole sentence groups and emit an exact membership manifest."""

    source = list(examples)
    source_data_hash = hash_examples(source)
    example_ids, group_ids, content_hashes, source_indices = _identity_maps(
        source, task_name
    )
    folds, diagnostics = grouped_stratified_split(
        source,
        dict(fractions),
        seed=seed,
        task_name=task_name,
        group_mode="auto",
    )
    manifest: list[dict] = []
    subset_hashes: dict[str, str] = {}
    for subset, rows in folds.items():
        ids = [example_ids[id(example)] for example in rows]
        subset_hashes[subset] = _membership_hash(ids)
        for position, example in enumerate(rows):
            manifest.append(
                {
                    "subset": subset,
                    "position_in_subset": position,
                    "source_index": source_indices[id(example)],
                    "example_id": example_ids[id(example)],
                    "group_id": group_ids[id(example)],
                    "content_hash": content_hashes[id(example)],
                    "zc": int(example.zc),
                    "ze": int(example.ze),
                    "source_data_hash": source_data_hash,
                    "split_seed": seed,
                    "subset_hash": subset_hashes[subset],
                }
            )

    owner: dict[str, str] = {}
    for row in manifest:
        previous = owner.setdefault(row["group_id"], row["subset"])
        if previous != row["subset"]:
            raise AssertionError(
                f"group {row['group_id']} crossed {previous}/{row['subset']}"
            )
    return GroupedSubdivision(
        task_name=task_name,
        seed=seed,
        source_data_hash=source_data_hash,
        folds=folds,
        manifest=manifest,
        subset_hashes=subset_hashes,
        diagnostics=diagnostics,
    )


def subdivide_phase2_intervention(
    examples: Sequence[Example], *, seed: int, task_name: str
) -> GroupedSubdivision:
    return subdivide_grouped(
        examples,
        CONSTRUCT_SUBDIVISION_FRACTIONS,
        seed=seed,
        task_name=task_name,
    )


@dataclass(frozen=True)
class CacheSelection:
    path: Path
    provenance_path: Path
    model_id: str
    task: str
    layer: int
    tag: str
    data_hash: str
    cache_sha256: str
    provenance: dict


def _resolve_cache_directory(cache_root: Path, model_id: str, task: str) -> Path:
    expected_name = f"{model_id.replace('/', '_')}_{task}"
    if cache_root.name == expected_name:
        directory = cache_root
    else:
        directory = cache_root / expected_name
    if not directory.is_dir():
        raise CacheSelectionError(
            f"cache directory for model={model_id!r}, task={task!r} does not exist: "
            f"{directory}"
        )
    return directory


def select_representation_cache(
    cache_root: Path | str,
    *,
    model_id: str,
    task: str,
    layer: int,
    tag: str,
    expected_data_hash: str,
    expected_cache_sha256: str | None = None,
) -> CacheSelection:
    """Select one cache by exact provenance; never fall back to glob order."""

    directory = _resolve_cache_directory(Path(cache_root), model_id, task)
    candidates = sorted(directory.glob(f"*_{tag}_L{int(layer)}_*.pt"))
    exact: list[CacheSelection] = []
    available_hashes: set[str] = set()
    rejected: list[str] = []
    for path in candidates:
        provenance_path = path.with_suffix(".json")
        if not provenance_path.is_file():
            rejected.append(f"{path.name}: missing provenance")
            continue
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            rejected.append(f"{path.name}: invalid provenance ({exc})")
            continue
        candidate_hash = str(provenance.get("data_hash", ""))
        if candidate_hash:
            available_hashes.add(candidate_hash)
        if provenance.get("model") != model_id:
            continue
        if int(provenance.get("layer", -1)) != int(layer):
            continue
        if provenance.get("cache_tag") != tag:
            continue
        if candidate_hash != expected_data_hash:
            continue
        if not path.stem.endswith(f"_{expected_data_hash}"):
            rejected.append(f"{path.name}: filename/provenance data hash mismatch")
            continue
        cache_sha256 = _sha256_file(path)
        recorded_cache_sha = provenance.get("cache_sha256")
        if recorded_cache_sha is not None and recorded_cache_sha != cache_sha256:
            rejected.append(f"{path.name}: recorded cache SHA-256 mismatch")
            continue
        if expected_cache_sha256 is not None and expected_cache_sha256 != cache_sha256:
            rejected.append(f"{path.name}: expected cache SHA-256 mismatch")
            continue
        exact.append(
            CacheSelection(
                path=path,
                provenance_path=provenance_path,
                model_id=model_id,
                task=task,
                layer=int(layer),
                tag=tag,
                data_hash=expected_data_hash,
                cache_sha256=cache_sha256,
                provenance=provenance,
            )
        )

    if len(exact) > 1:
        names = ", ".join(selection.path.name for selection in exact)
        raise CacheSelectionError(
            f"ambiguous cache selection for model={model_id}, task={task}, "
            f"layer={layer}, tag={tag}, data_hash={expected_data_hash}: {names}"
        )
    if not exact:
        details = "; ".join(rejected) if rejected else "none"
        raise CacheSelectionError(
            f"no exact cache for model={model_id}, task={task}, layer={layer}, "
            f"tag={tag}, expected data hash={expected_data_hash}; available data "
            f"hashes={sorted(available_hashes)}; rejected={details}"
        )
    return exact[0]


@dataclass(frozen=True)
class ValidatedCache:
    selection: CacheSelection
    X: torch.Tensor
    zc: torch.Tensor
    ze: torch.Tensor
    group_ids: tuple[str, ...] | None
    n_examples: int
    hidden_size: int
    mean_feature_variance: float
    class_conditioned_variance: dict[int, float]
    target_class_conditioned_variance: dict[int, float]
    control_class_conditioned_variance: dict[int, float]
    legacy_dtype_semantics: bool


def _validate_expected_labels(
    actual: torch.Tensor, expected, *, label_name: str
) -> None:
    if expected is None:
        return
    wanted = torch.as_tensor(expected).detach().cpu().reshape(-1).long()
    if not torch.equal(actual, wanted):
        raise CacheValidationError(f"{label_name} labels do not match reconstructed split")


def _class_conditioned_variance(
    X: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_name: str,
    minimum_variance: float,
) -> dict[int, float]:
    result: dict[int, float] = {}
    for label in sorted(int(value) for value in torch.unique(labels).tolist()):
        class_rows = X[labels == label]
        if len(class_rows) < 2:
            raise CacheValidationError(
                f"{label_name} class {label} has fewer than two representations"
            )
        variance = float(class_rows.var(dim=0, unbiased=False).mean().item())
        if not math.isfinite(variance) or variance <= minimum_variance:
            raise CacheValidationError(
                f"{label_name} class {label} representations are collapsed"
            )
        result[label] = variance
    return result


def load_validated_cache(
    selection: CacheSelection,
    *,
    expected_zc=None,
    expected_ze=None,
    expected_group_ids: Sequence[str] | None = None,
    minimum_variance: float = 1e-12,
) -> ValidatedCache:
    """Load a selected cache and enforce shape, label, finite, and variance gates."""

    try:
        payload = torch.load(selection.path, map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - exact torch errors vary by version
        raise CacheValidationError(f"failed to load {selection.path}: {exc}") from exc
    if not isinstance(payload, dict) or not {"X", "zc", "ze"}.issubset(payload):
        raise CacheValidationError("cache payload must contain X, zc, and ze tensors")
    X = torch.as_tensor(payload["X"]).detach().cpu()
    zc = torch.as_tensor(payload["zc"]).detach().cpu().reshape(-1).long()
    ze = torch.as_tensor(payload["ze"]).detach().cpu().reshape(-1).long()
    if X.ndim != 2:
        raise CacheValidationError(f"X must be rank two, got shape {tuple(X.shape)}")
    n_examples, hidden_size = X.shape
    if n_examples == 0 or hidden_size == 0:
        raise CacheValidationError("cache representations must be nonempty")
    if len(zc) != n_examples or len(ze) != n_examples:
        raise CacheValidationError("representation and label row counts differ")
    if expected_group_ids is not None and len(expected_group_ids) != n_examples:
        raise CacheValidationError("representation and group-ID row counts differ")
    if int(selection.provenance.get("n_examples", -1)) != n_examples:
        raise CacheValidationError("cache row count disagrees with provenance")
    if int(selection.provenance.get("hidden_size", -1)) != hidden_size:
        raise CacheValidationError("cache hidden size disagrees with provenance")
    representation_dtype = selection.provenance.get("representation_dtype")
    recorded_dtype = (
        representation_dtype
        if representation_dtype is not None
        else selection.provenance.get("dtype")
    )
    legacy_dtype_semantics = bool(
        representation_dtype is None
        and str(X.dtype) == "torch.float32"
        and recorded_dtype in {"torch.bfloat16", "torch.float16"}
    )
    if recorded_dtype != str(X.dtype) and not legacy_dtype_semantics:
        raise CacheValidationError("cache dtype disagrees with provenance")
    if selection.provenance.get("extraction_rule") != "last non-padding input token":
        raise CacheValidationError("cache does not identify the corrected extraction path")
    if not torch.isfinite(X).all():
        raise CacheValidationError("cache contains NaN or infinite representations")
    if not torch.all((zc == 0) | (zc == 1)) or not torch.all((ze == 0) | (ze == 1)):
        raise CacheValidationError("cache labels must be binary")
    _validate_expected_labels(zc, expected_zc, label_name="target")
    _validate_expected_labels(ze, expected_ze, label_name="control")

    X_float = X.float()
    mean_feature_variance = float(X_float.var(dim=0, unbiased=False).mean().item())
    if not math.isfinite(mean_feature_variance) or mean_feature_variance <= minimum_variance:
        raise CacheValidationError("cache representations are collapsed")
    target_class_variance = _class_conditioned_variance(
        X_float,
        zc,
        label_name="target",
        minimum_variance=minimum_variance,
    )
    control_class_variance = _class_conditioned_variance(
        X_float,
        ze,
        label_name="control",
        minimum_variance=minimum_variance,
    )

    return ValidatedCache(
        selection=selection,
        X=X,
        zc=zc,
        ze=ze,
        group_ids=(tuple(expected_group_ids) if expected_group_ids is not None else None),
        n_examples=n_examples,
        hidden_size=hidden_size,
        mean_feature_variance=mean_feature_variance,
        class_conditioned_variance=target_class_variance,
        target_class_conditioned_variance=target_class_variance,
        control_class_conditioned_variance=control_class_variance,
        legacy_dtype_semantics=legacy_dtype_semantics,
    )
def validate_cross_layer_cache_identity(caches: Mapping[int, ValidatedCache]) -> None:
    """Require one exact split and label order at every requested layer."""

    if not caches:
        raise CacheValidationError("no layer caches supplied")
    first_layer = min(caches)
    reference = caches[first_layer]
    for layer, cache in sorted(caches.items()):
        if cache.selection.data_hash != reference.selection.data_hash:
            raise CacheValidationError(
                f"layer {layer} data hash differs from layer {first_layer}"
            )
        if not torch.equal(cache.zc, reference.zc) or not torch.equal(cache.ze, reference.ze):
            raise CacheValidationError(
                f"layer {layer} label order differs from layer {first_layer}"
            )
        if cache.group_ids != reference.group_ids:
            raise CacheValidationError(
                f"layer {layer} group IDs differ from layer {first_layer}"
            )


__all__ = [
    "CONSTRUCT_SUBDIVISION_FRACTIONS",
    "PHASE2_FRACTIONS",
    "CacheSelection",
    "CacheSelectionError",
    "CacheValidationError",
    "GroupedSubdivision",
    "Phase2Reconstruction",
    "ValidatedCache",
    "load_validated_cache",
    "reconstruct_phase2_folds",
    "select_representation_cache",
    "subdivide_grouped",
    "subdivide_phase2_intervention",
    "validate_cross_layer_cache_identity",
]
