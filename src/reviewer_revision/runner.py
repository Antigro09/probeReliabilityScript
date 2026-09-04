"""End-to-end orchestration for the locked reviewer-revision experiment."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch

from src.extraction import (
    ModelBundle,
    _validate_extraction_position,
    extract_all_layers,
    load_model,
)
from src.probes import LinearProbe, MKAProbe, MLPProbe, ProbeTrainConfig, train_probe
from src.repro import set_seed
from src.tasks import Example
from src.ws5_repaired import train_independent_evaluators

from .analysis import (
    materialize_rows,
    summarize_construct_check,
    summarize_epsilon_sweep,
    summarize_matched_split,
)
from .artifacts import (
    RunContext,
    atomic_torch_save,
    atomic_write_json,
    atomic_write_via,
    canonical_json,
    sanitize_manifest_payload,
    sha256_file,
    sha256_json,
    sha256_tensor,
)
from .config import RevisionConfig
from .data import (
    GroupedSubdivision,
    Phase2Reconstruction,
    load_validated_cache,
    reconstruct_phase2_folds,
    select_representation_cache,
    subdivide_grouped,
    subdivide_phase2_intervention,
    validate_cross_layer_cache_identity,
)
from .experiments import (
    AttackerEvaluatorPair,
    ProbePair,
    alterrep_edit,
    assert_disjoint_named_groups,
    attack_edit,
    candidate_rank_one_direction,
    evaluate_fixed_evaluator_edit,
    pgd_step_size,
    rank_one_projection_edit,
    realized_linf_norm,
    reference_edit,
    score_conditions_for_edit,
    train_attacker_evaluator_pair,
    train_probe_pair,
    validate_archived_baseline,
    validate_linf_edit,
    validate_retrained_baseline,
)
from .extension_config import ConstructCell
from .scoring import binary_metrics_from_logits, compute_damage_score
from .training import (
    DecoderSpec,
    evaluate_logits,
    select_linear_hyperparameters,
    state_dict_sha256,
    train_deterministic_probe,
)

LOGGER = logging.getLogger("reviewer_revision")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = PROJECT_ROOT / "cache" / "benchmark_v2"
ARCHIVE_PATH = (
    PROJECT_ROOT
    / "assets"
    / "reviewer_revision"
    / "attacker_evaluator_reference.json"
)
MANUSCRIPT_TEMPLATE_PATH = (
    PROJECT_ROOT / "assets" / "reviewer_revision" / "main_revised_prepatch.tex"
)
MANUSCRIPT_TEMPLATE_TEXT_SHA256 = (
    "2711f7237be32b4d5cb86d5db02425606988760c8b278009f59e01c2a8a09cf1"
)

_PILOT_CONSTRUCT_CELL = ConstructCell(
    "qwen",
    "Qwen/Qwen2.5-1.5B",
    "sst2",
    14,
)

SOURCE_INPUTS = {
    "main_revised.tex": {
        "attachment_name": "1-main_revised.tex",
        "source_sha256": "b51237436cd595ac98a4d818b8eeb720716eb1f569c3fbaf1fd0ad393096ecbe",
        "repository_path": "assets/reviewer_revision/main_revised_prepatch.tex",
    },
    "revision_experiment_spec.yaml": {
        "attachment_name": "2-revision_experiment_spec.yaml",
        "source_sha256": "1b876ab9901e5c196fca386cac221f3c76f4cb385ffd520707204ae2d4053bb8",
    },
    "references.bib": {
        "attachment_name": "3-references.bib",
        "source_sha256": "33d921b0548159357e63f2bf0dcbde03c3210782dfaa13769cedbc63ae007e97",
    },
}


class _TeeStream:
    """Mirror text writes to the interactive stream and the immutable run log."""

    def __init__(self, primary: Any, log_handle: Any) -> None:
        self.primary = primary
        self.log_handle = log_handle
        self.encoding = getattr(primary, "encoding", "utf-8")

    def write(self, value: str) -> int:
        written = self.primary.write(value)
        self.log_handle.write(value)
        self.log_handle.flush()
        return len(value) if written is None else int(written)

    def flush(self) -> None:
        self.primary.flush()
        self.log_handle.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.primary, "isatty", lambda: False)())

    def fileno(self) -> int:
        return int(self.primary.fileno())


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def _current_commit() -> str:
    return _git("rev-parse", "HEAD")


def _starting_commit() -> str:
    try:
        return _git("merge-base", "HEAD", "main")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return _current_commit()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def resolve_resume_directory(
    output_root: str | Path,
    *,
    config_hash: str,
) -> Path:
    """Find the newest unlocked run matching the exact locked config."""

    root = Path(output_root)
    if (root / "run_manifest.json").is_file():
        candidates = [root]
    else:
        candidates = sorted(
            (path for path in root.glob("*") if (path / "run_manifest.json").is_file()),
            reverse=True,
        ) if root.exists() else []
    for candidate in candidates:
        try:
            manifest = json.loads((candidate / "run_manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("config_hash") != config_hash:
            continue
        lock_path = candidate / ".run.lock"
        if lock_path.exists():
            try:
                lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
                owner_pid = int(lock_payload["pid"])
                owner_is_alive = owner_pid > 0 and psutil.pid_exists(owner_pid)
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                owner_is_alive = False
            if owner_is_alive:
                continue
            recovered_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            stale_path = candidate / f".stale-run-lock-{recovered_at}.json"
            try:
                os.replace(lock_path, stale_path)
            except FileNotFoundError:
                # A closing owner can remove its lock between our read and rename.
                pass
            except OSError:
                # Another process may have recovered or reacquired the run first.
                continue
        return candidate
    raise FileNotFoundError(
        f"no resumable run under {root} matches config hash {config_hash}"
    )


def _state_payload(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def save_pair_checkpoint(
    path: str | Path,
    pair: AttackerEvaluatorPair,
    *,
    metadata: Mapping[str, Any],
) -> Path:
    input_dim = int(pair.attacker.target_probe.linear.in_features)
    payload = {
        "schema_version": 1,
        "input_dim": input_dim,
        "pair_seed": pair.pair_seed,
        "metadata": dict(metadata),
        "attacker_target": _state_payload(pair.attacker.target_probe),
        "attacker_control": _state_payload(pair.attacker.control_probe),
        "evaluator_target": _state_payload(pair.evaluator.target_probe),
        "evaluator_control": _state_payload(pair.evaluator.control_probe),
        "hashes": {
            "attacker_target": pair.attacker.target_checkpoint_hash,
            "attacker_control": pair.attacker.control_checkpoint_hash,
            "evaluator_target": pair.evaluator.target_checkpoint_hash,
            "evaluator_control": pair.evaluator.control_checkpoint_hash,
        },
    }
    return atomic_torch_save(path, payload)


def _probe_from_state(
    state: Mapping[str, torch.Tensor],
    input_dim: int,
    device: torch.device,
) -> tuple[LinearProbe, str]:
    probe = LinearProbe(input_dim).to(device)
    probe.load_state_dict(state)
    probe.eval()
    state_cpu = {name: tensor.detach().cpu() for name, tensor in state.items()}
    return probe, state_dict_sha256(state_cpu)


def load_pair_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[AttackerEvaluatorPair, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1:
        raise ValueError("pair checkpoint schema mismatch")
    input_dim = int(payload["input_dim"])
    at, at_hash = _probe_from_state(payload["attacker_target"], input_dim, device)
    ac, ac_hash = _probe_from_state(payload["attacker_control"], input_dim, device)
    et, et_hash = _probe_from_state(payload["evaluator_target"], input_dim, device)
    ec, ec_hash = _probe_from_state(payload["evaluator_control"], input_dim, device)
    hashes = payload["hashes"]
    actual = {
        "attacker_target": at_hash,
        "attacker_control": ac_hash,
        "evaluator_target": et_hash,
        "evaluator_control": ec_hash,
    }
    if actual != hashes:
        raise ValueError("pair checkpoint hash validation failed")
    seed = int(payload["pair_seed"])
    pair = AttackerEvaluatorPair(
        attacker=ProbePair(at, ac, at_hash, ac_hash, 10_000 + seed),
        evaluator=ProbePair(et, ec, et_hash, ec_hash, 20_000 + seed),
        pair_seed=seed,
    )
    return pair, dict(payload.get("metadata", {}))


def _data_paths(task: str) -> list[Path]:
    if task == "sva":
        return [
            PROJECT_ROOT / "data" / "numpred.train",
            PROJECT_ROOT / "data" / "numpred.val",
            PROJECT_ROOT / "data" / "numpred.test",
        ]
    if task == "sst2":
        # The task loader uses the locally cached HF dataset when this optional
        # reviewer bundle path is absent.
        return [PROJECT_ROOT / "data" / "sst2.tsv"]
    raise ValueError(f"unsupported revision task: {task}")


def _reconstruct_tasks(config: RevisionConfig) -> dict[str, Phase2Reconstruction]:
    return {
        task: reconstruct_phase2_folds(
            task,
            _data_paths(task),
            max_examples=30_000,
            seed=42,
        )
        for task in config.tasks
    }


def _manifest_subset_rows(
    reconstruction: Phase2Reconstruction, subset: str
) -> list[dict[str, Any]]:
    return sorted(
        (row for row in reconstruction.manifest if row["subset"] == subset),
        key=lambda row: row["position_in_subset"],
    )


def _group_ids(reconstruction: Phase2Reconstruction, subset: str) -> tuple[str, ...]:
    return tuple(row["group_id"] for row in _manifest_subset_rows(reconstruction, subset))


def _subdivision_indices(
    source: list[Example], subdivision: GroupedSubdivision
) -> dict[str, torch.Tensor]:
    positions = {id(example): index for index, example in enumerate(source)}
    indices = {
        subset: torch.tensor([positions[id(example)] for example in examples], dtype=torch.long)
        for subset, examples in subdivision.folds.items()
    }
    flattened = torch.cat(list(indices.values()))
    if len(flattened) != len(source) or len(torch.unique(flattened)) != len(source):
        raise ValueError("subdivision indices do not form an exact partition")
    return indices


def _three_way_scoring_split(
    reconstruction: Phase2Reconstruction,
    *,
    seed: int,
) -> tuple[GroupedSubdivision, dict[str, torch.Tensor]]:
    source = reconstruction.folds["intervention"]
    subdivision = subdivide_grouped(
        source,
        {
            "attacker_train": 1 / 3,
            "split_evaluator_train": 1 / 3,
            "score": 1 / 3,
        },
        seed=seed,
        task_name=reconstruction.task_name,
    )
    return subdivision, _subdivision_indices(source, subdivision)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment_report(device: torch.device) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    # psutil's Windows extension does not accept pathlib.Path even though the
    # POSIX implementation does.  Normalize at the cross-platform boundary.
    disk = psutil.disk_usage(str(PROJECT_ROOT))
    requirements_path = PROJECT_ROOT / "requirements.txt"
    installed_distributions = sorted(
        (
            str(distribution.metadata.get("Name") or distribution.name).lower(),
            str(distribution.version),
        )
        for distribution in importlib.metadata.distributions()
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "available_memory_bytes": int(memory.available),
        "total_memory_bytes": int(memory.total),
        "disk_free_bytes": int(disk.free),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "packages": {
            name: _package_version(name)
            for name in ("transformers", "scipy", "scikit-learn", "pandas", "pyarrow", "PyYAML")
        },
        "dependency_lock": {
            "requirements_file": requirements_path.name,
            "requirements_sha256": sha256_file(requirements_path),
            "requirements_bytes": requirements_path.stat().st_size,
            "installed_distribution_count": len(installed_distributions),
            "installed_distributions_sha256": sha256_json(installed_distributions),
        },
        "selected_device": str(device),
        "device_protocol": {
            "transformer_extraction": str(device),
            "candidate_mlp_and_jacobian": str(device),
            "candidate_linear_and_mka": "cpu",
            "linear_attackers_and_evaluators": "cpu",
            "fresh_decoders": "cpu",
            "statistics": "cpu",
        },
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "cuda_available_recorded_only": bool(torch.cuda.is_available()),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "environment_variables": {
            "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
        },
    }


def _input_manifest() -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for filename, provenance in SOURCE_INPUTS.items():
        path = PROJECT_ROOT / provenance.get("repository_path", filename)
        inputs[filename] = {
            **provenance,
            "repository_copy_sha256": sha256_file(path),
            "repository_bytes": path.stat().st_size,
        }
    return inputs


_TAG_TO_SUBSET = {
    "cand": "candidate",
    "eval": "evaluator",
    "inter": "intervention",
    "test": "test",
}


def _load_cache(
    *,
    model_id: str,
    task: str,
    layer: int,
    tag: str,
    reconstruction: Phase2Reconstruction,
    expected_cache_sha256: str | None = None,
):
    subset = _TAG_TO_SUBSET[tag]
    examples = reconstruction.folds[subset]
    selection = select_representation_cache(
        CACHE_ROOT,
        model_id=model_id,
        task=task,
        layer=layer,
        tag=tag,
        expected_data_hash=reconstruction.fold_hashes[subset],
        expected_cache_sha256=expected_cache_sha256,
    )
    return load_validated_cache(
        selection,
        expected_zc=[example.zc for example in examples],
        expected_ze=[example.ze for example in examples],
        expected_group_ids=_group_ids(reconstruction, subset),
    )


def _preflight_cache_pins(context: RunContext) -> dict[tuple[str, str, int, str], str]:
    report_path = context.run_dir / "preflight_report.json"
    if not report_path.is_file():
        raise RuntimeError("preflight cache pins are unavailable")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pins: dict[tuple[str, str, int, str], str] = {}
    for row in report.get("caches", []):
        key = (
            str(row["model_id"]),
            str(row["task"]),
            int(row["layer"]),
            str(row["tag"]),
        )
        if key in pins:
            raise RuntimeError(f"duplicate preflight cache pin: {key}")
        pins[key] = str(row["cache_sha256"])
    return pins


def _run_existing_suite() -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q"]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=3600,
    )
    elapsed = time.perf_counter() - started
    combined = (completed.stdout + "\n" + completed.stderr).strip()
    match = re.search(r"(\d+) passed", combined)
    report = {
        "command": "python -m pytest -q",
        "exit_code": completed.returncode,
        "passed": int(match.group(1)) if match else None,
        "wall_seconds": elapsed,
        "output": combined,
    }
    if completed.returncode != 0:
        raise RuntimeError(f"preflight test suite failed:\n{combined}")
    return report


def _real_padding_check(reconstruction: Phase2Reconstruction) -> dict[str, Any]:
    from transformers import AutoTokenizer

    model_id = "google/gemma-2-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    samples = [
        example.sentence for example in reconstruction.folds["candidate"][:20]
    ]
    bundle = ModelBundle(
        name=model_id,
        model=torch.nn.Identity(),
        tokenizer=tokenizer,
        n_layers=26,
        hidden_size=2304,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    info = _validate_extraction_position(bundle, samples)
    if info["n_samples_checked"] != 20:
        raise RuntimeError("Gemma real-token padding check did not inspect 20 examples")
    info["padding_side"] = tokenizer.padding_side
    return info


_EXTRACTION_BATCH_SIZE = {
    "bert": 64,
    "gemma": 16,
    "gpt2": 64,
    "llama": 16,
    "pythia": 64,
    "qwen": 32,
}


def _flush_model_memory() -> None:
    gc.collect()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except RuntimeError:
            LOGGER.debug("MPS cache flush was unavailable", exc_info=True)


def _regenerate_cache_group(
    *,
    model: Mapping[str, Any],
    task: str,
    reconstruction: Phase2Reconstruction,
    requirements: Iterable[tuple[str, int]],
    extraction_device: torch.device,
    trigger: Exception,
) -> dict[str, Any]:
    """Regenerate only one exact model-task cache group after validation failure."""

    regeneration_started = time.perf_counter()
    model_id = str(model["hf_id"])
    safe_model = model_id.replace("/", "_")
    cache_directory = (CACHE_ROOT / f"{safe_model}_{task}").resolve()
    cache_root = CACHE_ROOT.resolve()
    if not cache_directory.is_relative_to(cache_root):
        raise RuntimeError("resolved cache regeneration directory escapes cache root")
    cache_directory.mkdir(parents=True, exist_ok=True)
    requirement_list = sorted(
        {(str(tag), int(layer)) for tag, layer in requirements}
    )
    quarantine_root = (
        CACHE_ROOT
        / "rejected_reviewer_revision"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        / f"{safe_model}_{task}"
    )
    quarantined: list[str] = []
    for tag, layer in requirement_list:
        subset = _TAG_TO_SUBSET[tag]
        data_hash = reconstruction.fold_hashes[subset]
        for candidate in (
            cache_directory / f"{safe_model}_{tag}_L{layer}_{data_hash}.pt",
            cache_directory / f"{safe_model}_{tag}_L{layer}_{data_hash}.json",
        ):
            resolved = candidate.resolve()
            if not resolved.is_relative_to(cache_root):
                raise RuntimeError("cache quarantine candidate escapes cache root")
            if candidate.is_file():
                quarantine_root.mkdir(parents=True, exist_ok=True)
                destination = quarantine_root / candidate.name
                shutil.move(str(candidate), str(destination))
                quarantined.append(_relative(destination, PROJECT_ROOT))

    LOGGER.warning(
        "regenerating corrected caches for %s/%s on %s after validation failure",
        model_id,
        task,
        extraction_device,
    )
    bundle = None
    attempts: list[dict[str, Any]] = []
    try:
        bundle = load_model(
            model_id,
            device=extraction_device,
            dtype=torch.bfloat16,
            trust_remote_code=False,
        )
        requirements_by_tag: dict[str, list[int]] = {}
        for tag, layer in requirement_list:
            requirements_by_tag.setdefault(tag, []).append(layer)
        for tag, layers in sorted(requirements_by_tag.items()):
            examples = reconstruction.folds[_TAG_TO_SUBSET[tag]]
            batch_size = int(_EXTRACTION_BATCH_SIZE[str(model["key"])])
            while True:
                attempt_started = time.perf_counter()
                try:
                    extract_all_layers(
                        bundle,
                        examples,
                        layers,
                        batch_size=batch_size,
                        max_length=256,
                        cache_dir=cache_directory,
                        cache_tag=tag,
                    )
                    attempts.append(
                        {
                            "tag": tag,
                            "layers": layers,
                            "batch_size": batch_size,
                            "wall_seconds": time.perf_counter() - attempt_started,
                            "status": "ok",
                        }
                    )
                    break
                except RuntimeError as error:
                    message = str(error).lower()
                    if "out of memory" not in message or batch_size <= 1:
                        raise
                    attempts.append(
                        {
                            "tag": tag,
                            "layers": layers,
                            "batch_size": batch_size,
                            "wall_seconds": time.perf_counter() - attempt_started,
                            "status": "oom_retry",
                        }
                    )
                    batch_size = max(1, batch_size // 2)
                    _flush_model_memory()
    finally:
        if bundle is not None:
            del bundle
        _flush_model_memory()
    return sanitize_manifest_payload(
        {
            "model_key": model["key"],
            "model_id": model_id,
            "task": task,
            "trigger": f"{type(trigger).__name__}: {trigger}",
            "extraction_device": str(extraction_device),
            "requirements": [list(item) for item in requirement_list],
            "quarantined_artifacts": quarantined,
            "attempts": attempts,
            "wall_seconds": time.perf_counter() - regeneration_started,
            "status": "regenerated_and_pending_revalidation",
        }
    )


def _requires_padding_fix_regeneration(model_key: str, cache: Any) -> bool:
    """Legacy left-padded Gemma caches cannot be certified by metadata alone."""

    return bool(
        model_key == "gemma"
        and cache.selection.provenance.get("extraction_code_version")
        != "last-nonpadding-mask-index-v2"
    )


def run_preflight(
    context: RunContext,
    config: RevisionConfig,
    *,
    device: torch.device,
) -> dict[str, Any]:
    LOGGER.info("preflight: running full repository test suite")
    tests = _run_existing_suite()
    reconstructions = _reconstruct_tasks(config)
    cache_report: list[dict[str, Any]] = []
    cache_recovery: list[dict[str, Any]] = []
    LOGGER.info("preflight: validating all required representation caches")
    for model in config.models:
        model_id = str(model["hf_id"])
        for task in config.tasks:
            reconstruction = reconstructions[task]
            requirements = [("inter", int(layer)) for layer in model["layers"]]
            if model["key"] == "qwen" and task == "sst2":
                requirements.extend((tag, 14) for tag in ("cand", "eval", "test"))

            def load_group(
                current_model_id: str = model_id,
                current_model_key: str = str(model["key"]),
                current_task: str = task,
                current_reconstruction: Phase2Reconstruction = reconstruction,
                current_requirements: tuple[tuple[str, int], ...] = tuple(requirements),
            ) -> dict[tuple[str, int], Any]:
                loaded: dict[tuple[str, int], Any] = {}
                for tag, layer in current_requirements:
                    cache = _load_cache(
                        model_id=current_model_id,
                        task=current_task,
                        layer=layer,
                        tag=tag,
                        reconstruction=current_reconstruction,
                    )
                    if _requires_padding_fix_regeneration(current_model_key, cache):
                        raise RuntimeError(
                            "Gemma cache predates corrected last-nonpadding extraction"
                        )
                    loaded[(tag, layer)] = cache
                return loaded

            try:
                group_caches = load_group()
            except Exception as error:  # noqa: BLE001 - exact cache recovery gate
                _flush_model_memory()
                recovery = _regenerate_cache_group(
                    model=model,
                    task=task,
                    reconstruction=reconstruction,
                    requirements=requirements,
                    extraction_device=device,
                    trigger=error,
                )
                group_caches = load_group()
                recovery["status"] = "regenerated_and_revalidated"
                recovery["validated_cache_sha256"] = {
                    f"{tag}:L{layer}": cache.selection.cache_sha256
                    for (tag, layer), cache in sorted(group_caches.items())
                }
                cache_recovery.append(recovery)

            intervention_layers = {
                layer: cache
                for (tag, layer), cache in group_caches.items()
                if tag == "inter"
            }
            validate_cross_layer_cache_identity(intervention_layers)
            for (tag, layer), cache in sorted(group_caches.items()):
                cache_report.append(
                    {
                        "model_key": model["key"],
                        "model_id": model_id,
                        "task": task,
                        "layer": layer,
                        "tag": tag,
                        "data_hash": cache.selection.data_hash,
                        "cache_sha256": cache.selection.cache_sha256,
                        "cache_path": _relative(cache.selection.path, PROJECT_ROOT),
                        "n_examples": cache.n_examples,
                        "hidden_size": cache.hidden_size,
                        "mean_feature_variance": cache.mean_feature_variance,
                        "class_conditioned_variance": cache.class_conditioned_variance,
                        "legacy_dtype_semantics": cache.legacy_dtype_semantics,
                        "extraction_code_version": cache.selection.provenance.get(
                            "extraction_code_version"
                        ),
                        "extraction_provenance_status": (
                            "corrected_v2"
                            if cache.selection.provenance.get("extraction_code_version")
                            == "last-nonpadding-mask-index-v2"
                            else "legacy_right_padding_compatible"
                        ),
                    }
                )
            del group_caches, intervention_layers
            _flush_model_memory()

    data_report = {
        task: {
            "n_loaded": reconstruction.n_loaded,
            "n_deduplicated": reconstruction.n_deduplicated,
            "n_assigned_to_archived_phase2_folds": reconstruction.n_assigned,
            "n_excluded_by_archived_phase2_four_cell_balance": reconstruction.n_excluded,
            "exclusion_reason": (
                "archived_phase2_four_cell_balance"
                if reconstruction.n_excluded
                else None
            ),
            "all_data_hash": reconstruction.all_data_hash,
            "fold_sizes": reconstruction.fold_sizes,
            "fold_hashes": reconstruction.fold_hashes,
        }
        for task, reconstruction in reconstructions.items()
    }
    padding = _real_padding_check(reconstructions["sva"])
    environment = _environment_report(device)
    atomic_write_json(context.run_dir / "environment.json", environment)
    report = {
        "status": "ok",
        "tests": tests,
        "inputs": _input_manifest(),
        "locked_device_config": {
            "transformer_extraction_device": config.raw["runtime"][
                "transformer_extraction_device"
            ],
            "linear_probe_device": config.raw["runtime"]["linear_probe_device"],
            "statistics_device": config.raw["runtime"]["statistics_device"],
            "resolved_accelerator_eligible_device": str(device),
            "extraction_fallback_recorded": bool(
                str(config.raw["runtime"]["transformer_extraction_device"])
                != str(device)
            ),
        },
        "data": data_report,
        "caches": cache_report,
        "cache_recovery": cache_recovery,
        "gemma_real_padding_check": padding,
        "scientific_gates": {
            "cache_count": len(cache_report),
            "all_intervention_layers_present": len(
                [row for row in cache_report if row["tag"] == "inter"]
            ) == 60,
            "same_split_ids_across_layers": True,
            "finite_and_noncollapsed": True,
        },
    }
    atomic_write_json(context.run_dir / "preflight_report.json", report)
    context.update_manifest({"preflight": {"status": "ok"}})
    LOGGER.info("preflight passed: %s tests, %s caches", tests["passed"], len(cache_report))
    return report


def _timed_linear_probe(
    X: torch.Tensor,
    labels: torch.Tensor,
    *,
    seed: int,
    device: torch.device,
    config: ProbeTrainConfig,
) -> tuple[LinearProbe, str, float]:
    set_seed(seed)
    probe = LinearProbe(X.shape[1]).to(device)
    started = time.perf_counter()
    train_probe(probe, X, labels, config, device)
    elapsed = time.perf_counter() - started
    return probe, state_dict_sha256(_state_payload(probe)), elapsed


def _benchmark_one_cell(
    *,
    model_key: str,
    model_id: str,
    task: str,
    layer: int,
    reconstruction: Phase2Reconstruction,
    split_seed: int,
    device: torch.device,
    serialization_path: Path,
    expected_cache_sha256: str,
) -> dict[str, Any]:
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    started = time.perf_counter()
    cache = _load_cache(
        model_id=model_id,
        task=task,
        layer=layer,
        tag="inter",
        reconstruction=reconstruction,
        expected_cache_sha256=expected_cache_sha256,
    )
    cache_load_seconds = time.perf_counter() - started
    peak_rss = max(peak_rss, process.memory_info().rss)
    subdivision, indices = _three_way_scoring_split(
        reconstruction, seed=split_seed
    )
    attacker_idx = indices["attacker_train"]
    evaluator_idx = indices["split_evaluator_train"]
    score_idx = indices["score"]
    training_config = ProbeTrainConfig()
    attacker_target, attacker_target_hash, attacker_target_seconds = _timed_linear_probe(
        cache.X[attacker_idx], cache.zc[attacker_idx], seed=10_000,
        device=device, config=training_config,
    )
    attacker_control, attacker_control_hash, attacker_control_seconds = _timed_linear_probe(
        cache.X[attacker_idx], cache.ze[attacker_idx], seed=10_001,
        device=device, config=training_config,
    )
    evaluator_target, evaluator_target_hash, evaluator_target_seconds = _timed_linear_probe(
        cache.X[evaluator_idx], cache.zc[evaluator_idx], seed=20_000,
        device=device, config=training_config,
    )
    evaluator_control, evaluator_control_hash, evaluator_control_seconds = _timed_linear_probe(
        cache.X[evaluator_idx], cache.ze[evaluator_idx], seed=20_001,
        device=device, config=training_config,
    )
    peak_rss = max(peak_rss, process.memory_info().rss)
    X_score = cache.X[score_idx]
    zc_score = cache.zc[score_idx]
    ze_score = cache.ze[score_idx]
    method_timings: dict[str, Any] = {}
    sample_rows: list[dict[str, Any]] = []
    for method in ("alterrep", "fgsm", "pgd"):
        attack_started = time.perf_counter()
        if method == "alterrep":
            edited = alterrep_edit(
                X_score, zc_score, attacker_target, device=device, alpha=1.0
            )
        else:
            edited = attack_edit(
                method, X_score, zc_score, attacker_target, device=device,
                epsilon=0.5, pgd_steps=10,
            )
        generation_seconds = time.perf_counter() - attack_started
        score_started = time.perf_counter()
        rows = score_conditions_for_edit(
            X_pre=X_score,
            X_post=edited,
            target_labels=zc_score,
            control_labels=ze_score,
            matched_target_probe=attacker_target,
            matched_control_probe=attacker_control,
            split_target_probe=evaluator_target,
            split_control_probe=evaluator_control,
            device=device,
            common={
                "model_key": model_key,
                "task": task,
                "layer": layer,
                "pair_seed": 0,
                "method": method,
            },
        )
        score_seconds = time.perf_counter() - score_started
        method_timings[method] = {
            "kind": "coupled",
            "generation_seconds": generation_seconds,
            "both_condition_scores_seconds": score_seconds,
            "realized_linf_norm": (
                realized_linf_norm(X_score, edited) if method in {"fgsm", "pgd"} else None
            ),
        }
        sample_rows.extend(rows)
        peak_rss = max(peak_rss, process.memory_info().rss)
    for method in ("inlp", "rlace"):
        generation_started = time.perf_counter()
        edited = reference_edit(
            method,
            X_score,
            zc_score,
            device=device,
            inlp_iterations=10,
            rlace_rank=1,
            rlace_steps=500,
        )
        generation_seconds = time.perf_counter() - generation_started
        score_started = time.perf_counter()
        reference_row = score_conditions_for_edit(
            X_pre=X_score,
            X_post=edited,
            target_labels=zc_score,
            control_labels=ze_score,
            matched_target_probe=evaluator_target,
            matched_control_probe=evaluator_control,
            split_target_probe=evaluator_target,
            split_control_probe=evaluator_control,
            device=device,
            common={
                "model_key": model_key,
                "task": task,
                "layer": layer,
                "pair_seed": 0,
                "method": method,
            },
        )[0]
        reference_row["condition"] = "reference"
        score_seconds = time.perf_counter() - score_started
        method_timings[method] = {
            "kind": "reference",
            "generation_seconds": generation_seconds,
            "reference_score_seconds": score_seconds,
            "realized_linf_norm": None,
        }
        sample_rows.append(reference_row)
        peak_rss = max(peak_rss, process.memory_info().rss)
    serialization_started = time.perf_counter()
    atomic_write_json(serialization_path, sample_rows)
    serialization_seconds = time.perf_counter() - serialization_started
    probe_training_seconds = (
        attacker_target_seconds
        + attacker_control_seconds
        + evaluator_target_seconds
        + evaluator_control_seconds
    )
    coupled_pair_seconds = sum(
        values["generation_seconds"] + values["both_condition_scores_seconds"]
        for values in method_timings.values()
        if values["kind"] == "coupled"
    )
    reference_generation_seconds = sum(
        values["generation_seconds"]
        for values in method_timings.values()
        if values["kind"] == "reference"
    )
    reference_pair_scoring_seconds = sum(
        values["reference_score_seconds"]
        for values in method_timings.values()
        if values["kind"] == "reference"
    )
    pair_seconds = (
        probe_training_seconds
        + coupled_pair_seconds
        + reference_generation_seconds
        + reference_pair_scoring_seconds
        + serialization_seconds
    )
    return {
        "model_key": model_key,
        "model_id": model_id,
        "task": task,
        "layer": layer,
        "cache_load_seconds": cache_load_seconds,
        "target_attacker_training_seconds": attacker_target_seconds,
        "control_attacker_training_seconds": attacker_control_seconds,
        "target_split_evaluator_training_seconds": evaluator_target_seconds,
        "control_split_evaluator_training_seconds": evaluator_control_seconds,
        "method_timings": method_timings,
        "probe_training_seconds": probe_training_seconds,
        "coupled_pair_seconds": coupled_pair_seconds,
        "reference_generation_seconds": reference_generation_seconds,
        "reference_pair_scoring_seconds": reference_pair_scoring_seconds,
        "serialization_seconds": serialization_seconds,
        "one_pair_seconds_excluding_cache_load": pair_seconds,
        "peak_process_rss_bytes": int(peak_rss),
        "peak_mps_allocation_bytes": (
            int(torch.mps.current_allocated_memory())
            if device.type == "mps" and hasattr(torch, "mps")
            else None
        ),
        "n_attacker_train": int(attacker_idx.numel()),
        "n_split_evaluator_train": int(evaluator_idx.numel()),
        "n_score": int(score_idx.numel()),
        "checkpoint_hashes": {
            "attacker_target": attacker_target_hash,
            "attacker_control": attacker_control_hash,
            "evaluator_target": evaluator_target_hash,
            "evaluator_control": evaluator_control_hash,
        },
        "split_hashes": subdivision.subset_hashes,
        "sample_serialized_bytes": serialization_path.stat().st_size,
    }


def _project_runtime_from_benchmarks(
    cells: list[dict[str, Any]],
    *,
    selected_cells: int,
    pair_seeds: int,
) -> dict[str, Any]:
    """Project exact locked work-unit counts from conservative measured components."""

    if not cells:
        raise ValueError("at least one benchmark cell is required")
    maxima = {
        name: max(float(row[name]) for row in cells)
        for name in (
            "cache_load_seconds",
            "probe_training_seconds",
            "coupled_pair_seconds",
            "reference_generation_seconds",
            "reference_pair_scoring_seconds",
            "serialization_seconds",
        )
    }
    pair_seconds_without_cell_shared_reference = (
        maxima["probe_training_seconds"]
        + maxima["coupled_pair_seconds"]
        + maxima["reference_pair_scoring_seconds"]
        + maxima["serialization_seconds"]
    )

    def experiment_a_seconds(cell_count: int) -> float:
        return (
            cell_count
            * (maxima["cache_load_seconds"] + maxima["reference_generation_seconds"])
            + cell_count * pair_seeds * pair_seconds_without_cell_shared_reference
        )

    epsilon_method_seconds = max(
        sum(
            float(row["method_timings"][method]["generation_seconds"])
            + float(row["method_timings"][method]["both_condition_scores_seconds"])
            for method in ("fgsm", "pgd")
        )
        for row in cells
    )
    epsilon_pair_seconds = (
        10 * epsilon_method_seconds + 20 * maxima["serialization_seconds"]
    )
    required_epsilon_pairs = 12 * pair_seeds
    optional_epsilon_pairs = selected_cells * pair_seeds
    construct_pair_equivalent_units = 65 + 520 + 48 + 10
    conservative_complete_pair_seconds = (
        pair_seconds_without_cell_shared_reference
        + maxima["reference_generation_seconds"]
    )
    return {
        "component_maxima_seconds": maxima,
        "pair_seconds_without_cell_shared_reference": pair_seconds_without_cell_shared_reference,
        "measured_complete_pair_seconds": conservative_complete_pair_seconds,
        "full_grid_pair_units": 60 * pair_seeds,
        "fallback_pair_units": 36 * pair_seeds,
        "required_middle_epsilon_pair_units": required_epsilon_pairs,
        "required_middle_epsilon_attack_invocations": required_epsilon_pairs * 2 * 10,
        "optional_all_layer_epsilon_pair_units": optional_epsilon_pairs,
        "optional_all_layer_epsilon_attack_invocations": optional_epsilon_pairs * 2 * 10,
        "full_grid_seconds": experiment_a_seconds(60),
        "fallback_grid_seconds": experiment_a_seconds(36),
        "required_middle_epsilon_seconds": required_epsilon_pairs * epsilon_pair_seconds,
        "optional_all_layer_epsilon_seconds": optional_epsilon_pairs * epsilon_pair_seconds,
        "construct_check_seconds": construct_pair_equivalent_units
        * conservative_complete_pair_seconds,
        "construct_projection_basis": {
            "candidate_edits": 65,
            "fresh_decoder_fits": 520,
            "linear_hyperparameter_fits": 48,
            "attacker_evaluator_pair_equivalents": 10,
            "method": "conservative complete-pair-equivalent upper estimate",
            "used_for_scope_decision": False,
        },
    }


def _project_disk_usage(
    *,
    matched_cells: Iterable[Any],
    epsilon_cells: Iterable[Any],
    hidden_sizes: dict[tuple[str, str, int], int],
    score_sizes: dict[str, int],
    pair_seeds: int,
    epsilon_count: int,
    sample_bytes: int,
    construct_reserve_bytes: int,
) -> dict[str, Any]:
    """Return auditable byte accounting for one proposed pipeline scope."""

    epsilon_cells = tuple(epsilon_cells)
    epsilon_cell_keys = {
        (cell.model_key, cell.task, int(cell.layer)) for cell in epsilon_cells
    }
    matched_rows: list[dict[str, Any]] = []
    for cell in matched_cells:
        hidden_size = hidden_sizes[(cell.model_key, cell.task, int(cell.layer))]
        n_score = score_sizes[cell.task]
        checkpoint_bytes_per_pair = 32 * (hidden_size + 1)
        covered_by_epsilon = (
            cell.model_key,
            cell.task,
            int(cell.layer),
        ) in epsilon_cell_keys
        if covered_by_epsilon:
            # Epsilon=.5 attack deltas and both condition outputs share the
            # same immutable paths with Experiment B. A still adds AlterRep
            # direction recipes/outputs and two shared reference deltas/outputs.
            experiment_a_artifact_bytes = (
                pair_seeds * (4 * hidden_size + 448 * n_score)
                + 8 * n_score * hidden_size
            )
        else:
            # Three coupled methods (six condition outputs), two reference
            # outputs, two pair-specific attack deltas, and two shared
            # reference deltas. AlterRep stores a direction rather than a delta.
            experiment_a_artifact_bytes = (
                pair_seeds
                * (8 * n_score * hidden_size + 4 * hidden_size + 896 * n_score)
                + 8 * n_score * hidden_size
            )
        raw_bytes = pair_seeds * (checkpoint_bytes_per_pair + sample_bytes)
        raw_bytes += experiment_a_artifact_bytes
        matched_rows.append(
            {
                "model_key": cell.model_key,
                "task": cell.task,
                "layer": int(cell.layer),
                "hidden_size": hidden_size,
                "n_score": n_score,
                "pair_seeds": pair_seeds,
                "checkpoint_bytes_per_pair": checkpoint_bytes_per_pair,
                "sample_serialization_bytes_per_pair": sample_bytes,
                "epsilon_half_artifacts_shared_with_sweep": covered_by_epsilon,
                "experiment_a_artifact_bytes": experiment_a_artifact_bytes,
                "raw_bytes": raw_bytes,
            }
        )

    epsilon_rows: list[dict[str, Any]] = []
    for cell in epsilon_cells:
        hidden_size = hidden_sizes[(cell.model_key, cell.task, int(cell.layer))]
        n_score = score_sizes[cell.task]
        edit_units = pair_seeds * 2 * epsilon_count
        bytes_per_edit = n_score * (4 * hidden_size + 224)
        epsilon_rows.append(
            {
                "model_key": cell.model_key,
                "task": cell.task,
                "layer": int(cell.layer),
                "n_score": n_score,
                "hidden_size": hidden_size,
                "pair_seeds": pair_seeds,
                "methods": 2,
                "epsilon_count": epsilon_count,
                "edit_units": edit_units,
                "bytes_per_edit": bytes_per_edit,
                "raw_bytes": edit_units * bytes_per_edit,
            }
        )

    raw_artifact_bytes = sum(row["raw_bytes"] for row in matched_rows)
    raw_artifact_bytes += sum(row["raw_bytes"] for row in epsilon_rows)
    artifact_bytes_with_overhead = int(raw_artifact_bytes * 1.10)
    return {
        "matched_cells": matched_rows,
        "epsilon_cells": epsilon_rows,
        "raw_artifact_bytes": raw_artifact_bytes,
        "overhead_multiplier": 1.10,
        "artifact_bytes_with_overhead": artifact_bytes_with_overhead,
        "construct_reserve_bytes": construct_reserve_bytes,
        "total_bytes": artifact_bytes_with_overhead + construct_reserve_bytes,
        "epsilon_tensor_formula": (
            "n_score * (4*hidden_size + 224) bytes per attack edit; "
            "one lossless float32 delta plus two condition prediction payloads"
        ),
    }


def run_benchmark(
    context: RunContext,
    config: RevisionConfig,
    *,
    device: torch.device,
) -> dict[str, Any]:
    preflight = _require_ok_report(context, "preflight_report.json")
    cache_regeneration_seconds = sum(
        float(row.get("wall_seconds", 0.0)) for row in preflight.get("cache_recovery", [])
    )
    cache_pins = _preflight_cache_pins(context)
    reconstructions = _reconstruct_tasks(config)
    representatives = [
        ("bert", "google-bert/bert-base-uncased", "sva", 6),
        ("qwen", "Qwen/Qwen2.5-1.5B", "sst2", 14),
    ]
    cells: list[dict[str, Any]] = []
    benchmark_dir = context.run_dir / "benchmark"
    for model_key, model_id, task, layer in representatives:
        LOGGER.info("benchmarking one pair: %s/%s/L%s", model_key, task, layer)
        cells.append(
            _benchmark_one_cell(
                model_key=model_key,
                model_id=model_id,
                task=task,
                layer=layer,
                reconstruction=reconstructions[task],
                split_seed=int(config.raw["reproducibility"]["master_seed"]),
                device=torch.device("cpu"),
                serialization_path=benchmark_dir / f"{model_key}-{task}-L{layer}.json",
                expected_cache_sha256=cache_pins[(model_id, task, layer, "inter")],
            )
        )
    initial_projection = _project_runtime_from_benchmarks(
        cells, selected_cells=60, pair_seeds=len(config.pair_seeds)
    )
    full_hours = initial_projection["full_grid_seconds"] / 3600
    fallback_hours = initial_projection["fallback_grid_seconds"] / 3600
    epsilon_required_hours = (
        initial_projection["required_middle_epsilon_seconds"] / 3600
    )
    sample_bytes = max(row["sample_serialized_bytes"] for row in cells)
    hidden_sizes = {
        (row["model_key"], row["task"], int(row["layer"])): int(row["hidden_size"])
        for row in preflight["caches"]
        if row["tag"] == "inter"
    }
    score_sizes = {row["task"]: int(row["n_score"]) for row in cells}

    middle_cells = tuple(
        cell
        for cell in config.matched_split_cells("full")
        if cell.layer == config.model(cell.model_key)["middle_layer"]
    )
    construct_reserve_bytes = 8 * 1024**3
    full_disk = _project_disk_usage(
        matched_cells=config.matched_split_cells("full"),
        epsilon_cells=middle_cells,
        hidden_sizes=hidden_sizes,
        score_sizes=score_sizes,
        pair_seeds=len(config.pair_seeds),
        epsilon_count=len(config.epsilons),
        sample_bytes=sample_bytes,
        construct_reserve_bytes=construct_reserve_bytes,
    )
    full_projected_bytes = full_disk["total_bytes"]
    projected_disk_gb = full_projected_bytes / 1024**3
    full_allowed = (
        full_hours <= float(config.raw["runtime"]["full_grid_max_projected_hours"])
        and projected_disk_gb <= float(config.raw["runtime"]["maximum_projected_disk_gb"])
    )
    selected_grid = "full" if full_allowed else "fallback"
    if not full_allowed and fallback_hours > float(
        config.raw["runtime"]["full_grid_max_projected_hours"]
    ):
        raise RuntimeError(
            f"structured blocker: projected fallback runtime {fallback_hours:.2f} h exceeds 48 h"
        )
    projection = _project_runtime_from_benchmarks(
        cells,
        selected_cells=len(config.matched_split_cells(selected_grid)),
        pair_seeds=len(config.pair_seeds),
    )
    epsilon_all_layer_hours = projection["optional_all_layer_epsilon_seconds"] / 3600
    selected_cells = config.matched_split_cells(selected_grid)
    selected_all_layer_disk = _project_disk_usage(
        matched_cells=selected_cells,
        epsilon_cells=selected_cells,
        hidden_sizes=hidden_sizes,
        score_sizes=score_sizes,
        pair_seeds=len(config.pair_seeds),
        epsilon_count=len(config.epsilons),
        sample_bytes=sample_bytes,
        construct_reserve_bytes=construct_reserve_bytes,
    )
    selected_middle_disk = _project_disk_usage(
        matched_cells=selected_cells,
        epsilon_cells=middle_cells,
        hidden_sizes=hidden_sizes,
        score_sizes=score_sizes,
        pair_seeds=len(config.pair_seeds),
        epsilon_count=len(config.epsilons),
        sample_bytes=sample_bytes,
        construct_reserve_bytes=construct_reserve_bytes,
    )
    selected_total_disk_gb = selected_all_layer_disk["total_bytes"] / 1024**3
    epsilon_scope = (
        "all_selected_layers"
        if epsilon_all_layer_hours
        <= float(config.raw["runtime"]["epsilon_all_layers_max_projected_hours"])
        and selected_total_disk_gb
        <= float(config.raw["runtime"]["maximum_projected_disk_gb"])
        else "middle_only"
    )
    seconds_per_pair = projection["measured_complete_pair_seconds"]
    report = {
        "status": "ok",
        "representative_cells": cells,
        "conservative_one_pair_seconds": seconds_per_pair,
        "projections": {
            "full_grid_hours": full_hours,
            "fallback_grid_hours": fallback_hours,
            "required_middle_epsilon_hours": epsilon_required_hours,
            "optional_all_layer_epsilon_hours": epsilon_all_layer_hours,
            "construct_check_hours": projection["construct_check_seconds"] / 3600,
            "projected_disk_gb": projected_disk_gb,
            "selected_scope_total_disk_gb": selected_total_disk_gb,
            "cache_regeneration_hours_separate": cache_regeneration_seconds / 3600,
            "full_grid_plus_completed_cache_recovery_hours": (
                full_hours + cache_regeneration_seconds / 3600
            ),
            "fallback_grid_plus_completed_cache_recovery_hours": (
                fallback_hours + cache_regeneration_seconds / 3600
            ),
        },
        "work_unit_projection": projection,
        "disk_projection": {
            "full_required_pipeline_bytes": full_projected_bytes,
            "selected_all_layer_pipeline_bytes": selected_all_layer_disk["total_bytes"],
            "selected_middle_only_pipeline_bytes": selected_middle_disk["total_bytes"],
            "full_required_pipeline": full_disk,
            "selected_all_layer_pipeline": selected_all_layer_disk,
            "selected_middle_only_pipeline": selected_middle_disk,
        },
        "limits": {
            "full_grid_max_hours": config.raw["runtime"]["full_grid_max_projected_hours"],
            "epsilon_all_layers_max_hours": config.raw["runtime"]["epsilon_all_layers_max_projected_hours"],
            "maximum_disk_gb": config.raw["runtime"]["maximum_projected_disk_gb"],
        },
        "selected_grid": selected_grid,
        "selected_cells": len(config.matched_split_cells(selected_grid)),
        "epsilon_scope": epsilon_scope,
    }
    atomic_write_json(context.run_dir / "runtime_benchmark.json", report)
    context.update_manifest(
        {
            "benchmark": {
                "status": "ok",
                "selected_grid": selected_grid,
                "epsilon_scope": epsilon_scope,
                "one_pair_seconds": seconds_per_pair,
            }
        }
    )
    LOGGER.info(
        "benchmark selected %s grid (%s cells); one pair %.2fs",
        selected_grid,
        report["selected_cells"],
        seconds_per_pair,
    )
    return report


def run_reproduce_baseline(
    context: RunContext,
    config: RevisionConfig,
) -> dict[str, Any]:
    from scripts.ws4_attacker_evaluator import (
        DIRECTION_METHODS,
        _default_inter_cfg,
        _default_probe_cfg,
        attacker_evaluator_completeness,
    )

    archived_report = validate_archived_baseline(
        ARCHIVE_PATH, absolute_tolerance=1.0e-12
    )
    archived_payload = json.loads(ARCHIVE_PATH.read_text(encoding="utf-8"))
    current_cells: dict[str, Any] = {}
    cpu = torch.device("cpu")
    for cell_key in sorted(archived_payload["cells"]):
        archived_cell = archived_payload["cells"][cell_key]
        model_id = str(archived_cell["model"])
        task = str(archived_cell["task"])
        safe_model = model_id.replace("/", "_")
        cache_filename = re.split(r"[\\/]", str(archived_cell["cache_file"]))[-1]
        cache_path = CACHE_ROOT / f"{safe_model}_{task}" / cache_filename
        if not cache_path.is_file():
            raise RuntimeError(
                f"archived baseline cache is missing for {cell_key}: {cache_filename}"
            )
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if {"X", "zc", "ze"} - set(cached):
            raise RuntimeError(f"archived cache schema is invalid for {cell_key}")
        LOGGER.info("baseline rerun: %s using %s", cell_key, cache_filename)
        result = attacker_evaluator_completeness(
            cached["X"].float(),
            cached["zc"].long(),
            cached["ze"].long(),
            cpu,
            M=5,
            cfg=_default_probe_cfg(),
            inter_cfg=_default_inter_cfg(),
            floor=float(config.raw["shared_score_definition"]["denominator_floor"]),
            methods=DIRECTION_METHODS,
            controls=(),
            bootstrap=int(archived_payload["params"]["bootstrap"]),
            seed=int(archived_payload["params"]["seed"]),
        )
        result.update(
            {
                "model": model_id,
                "task": task,
                "cache_ref": _relative(cache_path, PROJECT_ROOT),
                "cache_sha256": sha256_file(cache_path),
                "layers_present": archived_cell["layers_present"],
            }
        )
        current_cells[cell_key] = result
        del cached
        gc.collect()

    current_payload = {
        "schema": "reviewer_revision_current_code_baseline_rerun_v1",
        "source_archive_sha256": archived_report["archive_sha256"],
        "device": "cpu",
        "probe_config": asdict(_default_probe_cfg()),
        "intervention_config": asdict(_default_inter_cfg()),
        "cells": current_cells,
    }
    current_path = context.run_dir / "baseline_current_12_cells.json"
    atomic_write_json(current_path, current_payload)
    try:
        report = validate_retrained_baseline(
            ARCHIVE_PATH,
            current_payload,
            aggregate_tolerance=0.01,
        )
    except Exception as error:
        failure_report = {
            "status": "failed",
            "failure_stage": "current_code_baseline_comparison",
            "failure_reason": f"{type(error).__name__}: {error}",
            "archive_sha256": archived_report["archive_sha256"],
            "current_rerun_ref": _relative(current_path, context.run_dir),
            "current_rerun_sha256": sha256_file(current_path),
        }
        atomic_write_json(context.run_dir / "baseline_reproduction.json", failure_report)
        context.update_manifest({"baseline_reproduction": failure_report})
        raise
    report.update(
        {
            "reproduction_mode": "deterministic_current_code_rerun_of_archived_splits",
            "archive_path": _relative(ARCHIVE_PATH, PROJECT_ROOT),
            "archive_sha256": archived_report["archive_sha256"],
            "current_rerun_ref": _relative(current_path, context.run_dir),
            "current_rerun_sha256": sha256_file(current_path),
            "checkpoint_availability": "archived pair checkpoints were not persisted",
            "scientific_device": "cpu_float32",
            "estimand_note": (
                "the archived deterministic three-way split is rerun before the "
                "new group-aware expansion"
            ),
        }
    )
    atomic_write_json(context.run_dir / "baseline_reproduction.json", report)
    context.update_manifest({"baseline_reproduction": {"status": "ok"}})
    LOGGER.info(
        "archived 12-cell gate reproduced: matched=%.15f split=%.15f gap=%.15f",
        report["current"]["matched"],
        report["current"]["split"],
        report["current"]["gap"],
    )
    return report


def _load_benchmark_report(context: RunContext) -> dict[str, Any]:
    path = context.run_dir / "runtime_benchmark.json"
    if not path.is_file():
        raise RuntimeError("runtime benchmark must pass before experiments")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "ok" or report.get("selected_grid") not in {"full", "fallback"}:
        raise RuntimeError("runtime benchmark report is invalid")
    return report


def _require_ok_report(context: RunContext, filename: str) -> dict[str, Any]:
    path = context.run_dir / filename
    if not path.is_file():
        raise RuntimeError(f"required prior gate is missing: {filename}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "ok":
        raise RuntimeError(f"required prior gate did not pass: {filename}")
    return report


def _checkpoint_path(
    context: RunContext,
    *,
    model_key: str,
    task: str,
    layer: int,
    pair_seed: int,
) -> Path:
    return (
        context.run_dir
        / "checkpoints"
        / "matched_split"
        / f"{model_key}-{task}-L{layer}-pair{pair_seed}.pt"
    )


def _load_or_train_pair(
    context: RunContext,
    *,
    model_key: str,
    task: str,
    layer: int,
    pair_seed: int,
    cache,
    indices: Mapping[str, torch.Tensor],
    split_hashes: Mapping[str, str],
    split_manifest_ref: str,
    split_manifest_sha256: str,
    device: torch.device,
) -> AttackerEvaluatorPair:
    path = _checkpoint_path(
        context,
        model_key=model_key,
        task=task,
        layer=layer,
        pair_seed=pair_seed,
    )
    expected_metadata = {
        "cell": [model_key, task, int(layer)],
        "pair_seed": int(pair_seed),
        "cache_sha256": cache.selection.cache_sha256,
        "split_hashes": dict(split_hashes),
        "split_manifest_ref": split_manifest_ref,
        "split_manifest_sha256": split_manifest_sha256,
    }
    if path.is_file():
        pair, metadata = load_pair_checkpoint(path, device=device)
        if metadata != expected_metadata:
            raise RuntimeError(f"checkpoint provenance mismatch: {_relative(path, context.run_dir)}")
        return pair
    pair = train_attacker_evaluator_pair(
        cache.X[indices["attacker_train"]],
        cache.zc[indices["attacker_train"]],
        cache.ze[indices["attacker_train"]],
        cache.X[indices["split_evaluator_train"]],
        cache.zc[indices["split_evaluator_train"]],
        cache.ze[indices["split_evaluator_train"]],
        pair_seed=pair_seed,
        device=device,
        config=ProbeTrainConfig(),
    )
    save_pair_checkpoint(path, pair, metadata=expected_metadata)
    return pair


def _base_row(
    context: RunContext,
    config: RevisionConfig,
    *,
    model: Mapping[str, Any],
    task: str,
    layer: int,
    pair_seed: int,
    method: str,
    cache,
    subdivision: GroupedSubdivision,
    indices: Mapping[str, torch.Tensor],
    pair: AttackerEvaluatorPair | None,
    split_manifest_ref: str,
    split_manifest_sha256: str,
) -> dict[str, Any]:
    layers = list(model["layers"])
    return {
        "run_id": context.run_id,
        "git_commit": _current_commit(),
        "config_hash": config.config_hash,
        "model_key": model["key"],
        "model_id": model["hf_id"],
        "task": task,
        "layer": int(layer),
        "depth_position": layers.index(layer) + 1,
        "pair_seed": int(pair_seed),
        "method": method,
        "failure_reason": None,
        "n_attacker_train": int(indices["attacker_train"].numel()),
        "n_split_evaluator_train": int(indices["split_evaluator_train"].numel()),
        "n_score": int(indices["score"].numel()),
        "attacker_group_hash": subdivision.subset_hashes["attacker_train"],
        "split_evaluator_group_hash": subdivision.subset_hashes["split_evaluator_train"],
        "score_group_hash": subdivision.subset_hashes["score"],
        "split_manifest_ref": split_manifest_ref,
        "split_manifest_sha256": split_manifest_sha256,
        "cache_hash": cache.selection.cache_sha256,
        "cache_data_hash": cache.selection.data_hash,
        "attacker_checkpoint_hash": (
            pair.attacker.target_checkpoint_hash if pair is not None else None
        ),
        "attacker_control_checkpoint_hash": (
            pair.attacker.control_checkpoint_hash if pair is not None else None
        ),
        "split_evaluator_checkpoint_hash": (
            pair.evaluator.target_checkpoint_hash if pair is not None else None
        ),
        "split_control_checkpoint_hash": (
            pair.evaluator.control_checkpoint_hash if pair is not None else None
        ),
    }


def _persist_scoring_split_manifest(
    context: RunContext,
    *,
    reconstruction: Phase2Reconstruction,
    subdivision: GroupedSubdivision,
    indices: Mapping[str, torch.Tensor],
    seed: int,
) -> tuple[str, str]:
    subsets: dict[str, Any] = {}
    for subset in ("attacker_train", "split_evaluator_train", "score"):
        rows = sorted(
            (row for row in subdivision.manifest if row["subset"] == subset),
            key=lambda row: row["position_in_subset"],
        )
        if len(rows) != int(indices[subset].numel()):
            raise RuntimeError(f"{subset} manifest length differs from index tensor")
        subsets[subset] = {
            "count": len(rows),
            "membership_sha256": subdivision.subset_hashes[subset],
            "ordered_source_indices": indices[subset].tolist(),
            "ordered_example_ids": [row["example_id"] for row in rows],
            "ordered_group_ids": [row["group_id"] for row in rows],
            "rows": rows,
        }
    payload = {
        "schema_version": 1,
        "task": reconstruction.task_name,
        "seed": int(seed),
        "source_fold": "phase2_intervention",
        "source_fold_hash": reconstruction.fold_hashes["intervention"],
        "source_all_data_hash": reconstruction.all_data_hash,
        "group_disjoint": True,
        "subdivision_diagnostics": subdivision.diagnostics,
        "subsets": subsets,
    }
    path = context.run_dir / "splits" / f"matched-split-{reconstruction.task_name}.json"
    _write_immutable_json(path, payload)
    return _relative(path, context.run_dir), sha256_file(path)


def _failure_rows(
    common: Mapping[str, Any],
    *,
    conditions: Iterable[str],
    error: Exception,
    failure_stage: str = "edit_generation_or_scoring",
    pre_metrics: Mapping[str, Mapping[str, float]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        condition_pre = (pre_metrics or {}).get(condition, {})
        rows.append(
            {
            **common,
            "condition": condition,
            "status": "failed",
            "failure_stage": failure_stage,
            "failure_reason": f"{type(error).__name__}: {error}",
            "edit_hash": None,
            "target_acc_pre": condition_pre.get("target_acc_pre"),
            "target_acc_post": None,
            "control_acc_pre": condition_pre.get("control_acc_pre"),
            "control_acc_post": None,
            "C": None,
            "S": None,
            "H": None,
            "C_raw": None,
            "S_raw": None,
            "realized_linf_norm": None,
            "wall_seconds": None,
            }
        )
    return rows


def _pre_edit_metrics(
    pair: AttackerEvaluatorPair,
    X: torch.Tensor,
    target_labels: torch.Tensor,
    control_labels: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for condition, target_probe, control_probe in (
        (
            "matched",
            pair.attacker.target_probe,
            pair.attacker.control_probe,
        ),
        (
            "split",
            pair.evaluator.target_probe,
            pair.evaluator.control_probe,
        ),
        (
            "reference",
            pair.evaluator.target_probe,
            pair.evaluator.control_probe,
        ),
    ):
        target = binary_metrics_from_logits(
            evaluate_logits(target_probe, X, device), target_labels
        )
        control = binary_metrics_from_logits(
            evaluate_logits(control_probe, X, device), control_labels
        )
        output[condition] = {
            "target_acc_pre": float(target["accuracy"]),
            "control_acc_pre": float(control["accuracy"]),
        }
    return output


def _normalize_score_status(row: dict[str, Any]) -> dict[str, Any]:
    """Preserve prespecified score-null statuses separately from hard failures."""

    score_status = row.get("status")
    reason = row.pop("reason", None)
    row["score_status"] = score_status
    if score_status == "ok":
        row["failure_reason"] = None
    else:
        row["status"] = score_status
        row["failure_reason"] = reason or f"damage score status: {score_status}"
        row["failure_stage"] = "scoring"
    return row


def _hard_failure_mask(frame: pd.DataFrame) -> pd.Series:
    """Return rows that reflect execution/integrity failures, not score nulls."""

    return frame["status"].astype(str).isin(("failed", "invalid"))


def _matched_edit_path(context: RunContext, key: tuple[Any, ...]) -> Path:
    return (
        context.run_dir
        / "arrays"
        / "matched_split"
        / "edits"
        / f"{sha256_json(list(key))}.pt"
    )


def _matched_array_path(context: RunContext, key: tuple[Any, ...]) -> Path:
    return (
        context.run_dir
        / "arrays"
        / "matched_split"
        / f"{sha256_json(list(key))}.pt"
    )


def _augment_matched_artifacts(
    context: RunContext,
    rows: list[dict[str, Any]],
    *,
    pair: AttackerEvaluatorPair,
    X_pre: torch.Tensor,
    X_post: torch.Tensor,
    target_labels: torch.Tensor,
    control_labels: torch.Tensor,
    example_ids: list[str],
    source_indices: list[int],
    split_name: str,
    source_cache_sha256: str,
    source_data_hash: str,
    method: str,
    device: torch.device,
) -> None:
    """Persist one shared edit and condition-specific per-example A outputs."""

    if len(rows) not in {1, 2} or len(example_ids) != len(X_pre):
        raise ValueError("matched/split artifact rows or example IDs are misaligned")
    if len(source_indices) != len(X_pre) or X_pre.shape != X_post.shape:
        raise ValueError("matched/split source indices or representation shapes differ")
    if method in {"fgsm", "pgd"}:
        _augment_epsilon_metrics(
            context,
            rows,
            pair=pair,
            X_pre=X_pre,
            X_post=X_post,
            target_labels=target_labels,
            control_labels=control_labels,
            example_ids=example_ids,
            split_name=split_name,
            source_cache_sha256=source_cache_sha256,
            source_data_hash=source_data_hash,
            source_indices=source_indices,
            epsilon=0.5,
            method=method,
            device=device,
        )
        for row in rows:
            row["artifact_scope"] = "experiment_a_and_epsilon_0.5_shared"
            row["source_indices_sha256"] = sha256_json(source_indices)
        return

    edit_seed: int | str = rows[0]["pair_seed"]
    if method in {"inlp", "rlace"}:
        edit_seed = "shared_across_pair_seeds"
    edit_key = (
        rows[0]["model_key"],
        rows[0]["task"],
        int(rows[0]["layer"]),
        edit_seed,
        method,
    )
    X_pre_cpu = X_pre.detach().cpu().float().contiguous()
    X_post_cpu = X_post.detach().cpu().float().contiguous()
    edit_hash = sha256_tensor(X_post_cpu)
    if any(row.get("edit_hash") != edit_hash for row in rows):
        raise RuntimeError("saved matched/split edit hash differs from scored tensor")
    recipe: dict[str, Any] = {
        "schema_version": 2,
        "key": list(edit_key),
        "method": method,
        "example_ids": list(example_ids),
        "source_indices": list(source_indices),
        "split_names": [split_name] * len(example_ids),
        "source_cache_sha256": source_cache_sha256,
        "source_data_hash": source_data_hash,
        "source_representation_dtype": str(X_pre.dtype),
        "output_dtype": str(X_post_cpu.dtype),
        "edit_hash": edit_hash,
        "implementation_ref": "src/interventions.py",
        "implementation_sha256": sha256_file(PROJECT_ROOT / "src" / "interventions.py"),
    }
    if method == "alterrep":
        weight = pair.attacker.target_probe.linear.weight.detach().cpu().float()
        direction = (weight[1] - weight[0]).contiguous()
        norm = direction.norm()
        if not torch.isfinite(norm) or float(norm) <= 0.0:
            raise RuntimeError("AlterRep attacker direction is degenerate")
        direction = (direction / (norm + 1.0e-9)).contiguous()
        signs = torch.where(target_labels.detach().cpu() == 1, -1.0, 1.0).unsqueeze(1)
        reconstructed = (X_pre_cpu + signs * direction.unsqueeze(0)).contiguous()
        maximum_error = float((reconstructed - X_post_cpu).abs().max().item())
        if maximum_error > 1.0e-6 or sha256_tensor(reconstructed) != edit_hash:
            raise RuntimeError(
                f"AlterRep lossless reconstruction failed: max_error={maximum_error}"
            )
        recipe.update(
            {
                "alpha": 1.0,
                "direction": direction,
                "direction_sha256": sha256_tensor(direction),
                "target_labels": target_labels.detach().cpu().long(),
                "reconstruction_rule": (
                    "X_post = X_pre + where(target_label == 1, -1, 1) * "
                    "alpha * normalized_attacker_direction"
                ),
                "reconstruction_max_abs_error": maximum_error,
            }
        )
    else:
        delta = (X_post_cpu - X_pre_cpu).contiguous()
        recipe.update(
            {
                "representation_delta": delta,
                "delta_sha256": sha256_tensor(delta),
                "reconstruction_rule": "X_post = X_pre + representation_delta",
            }
        )
    edit_path = _matched_edit_path(context, edit_key)
    _write_immutable_torch(edit_path, recipe)
    relative_edit = _relative(edit_path, context.run_dir)

    for row in rows:
        condition = str(row["condition"])
        if condition == "matched":
            target_probe = pair.attacker.target_probe
            control_probe = pair.attacker.control_probe
            checkpoint_hashes = {
                "target": pair.attacker.target_checkpoint_hash,
                "control": pair.attacker.control_checkpoint_hash,
            }
        else:
            target_probe = pair.evaluator.target_probe
            control_probe = pair.evaluator.control_probe
            checkpoint_hashes = {
                "target": pair.evaluator.target_checkpoint_hash,
                "control": pair.evaluator.control_checkpoint_hash,
            }
        target_logits_pre = evaluate_logits(target_probe, X_pre_cpu, device)
        target_logits_post = evaluate_logits(target_probe, X_post_cpu, device)
        control_logits_pre = evaluate_logits(control_probe, X_pre_cpu, device)
        control_logits_post = evaluate_logits(control_probe, X_post_cpu, device)
        target_pre_metrics = binary_metrics_from_logits(target_logits_pre, target_labels)
        target_post_metrics = binary_metrics_from_logits(target_logits_post, target_labels)
        control_pre_metrics = binary_metrics_from_logits(control_logits_pre, control_labels)
        control_post_metrics = binary_metrics_from_logits(control_logits_post, control_labels)
        row_key = (
            row["model_key"],
            row["task"],
            int(row["layer"]),
            int(row["pair_seed"]),
            method,
            condition,
        )
        array_path = _matched_array_path(context, row_key)
        _write_immutable_torch(
            array_path,
            {
                "schema_version": 2,
                "key": list(row_key),
                "example_ids": list(example_ids),
                "source_indices": list(source_indices),
                "split_names": [split_name] * len(example_ids),
                "target_labels": target_labels.detach().cpu().long(),
                "control_labels": control_labels.detach().cpu().long(),
                "source_cache_sha256": source_cache_sha256,
                "source_data_hash": source_data_hash,
                "edit_artifact_ref": relative_edit,
                "edit_hash": edit_hash,
                "evaluator_checkpoint_hashes": checkpoint_hashes,
                "target_logits_pre": target_logits_pre,
                "target_logits_post": target_logits_post,
                "target_probabilities_pre": torch.softmax(target_logits_pre.float(), dim=1),
                "target_probabilities_post": torch.softmax(target_logits_post.float(), dim=1),
                "target_predictions_pre": target_logits_pre.argmax(dim=1),
                "target_predictions_post": target_logits_post.argmax(dim=1),
                "control_logits_pre": control_logits_pre,
                "control_logits_post": control_logits_post,
                "control_probabilities_pre": torch.softmax(control_logits_pre.float(), dim=1),
                "control_probabilities_post": torch.softmax(control_logits_post.float(), dim=1),
                "control_predictions_pre": control_logits_pre.argmax(dim=1),
                "control_predictions_post": control_logits_post.argmax(dim=1),
                "confusion_matrices": {
                    "target_pre": target_pre_metrics["confusion_matrix"],
                    "target_post": target_post_metrics["confusion_matrix"],
                    "control_pre": control_pre_metrics["confusion_matrix"],
                    "control_post": control_post_metrics["confusion_matrix"],
                },
            },
        )
        relative_array = _relative(array_path, context.run_dir)
        row.update(
            {
                "edit_artifact_ref": relative_edit,
                "per_example_artifact_ref": relative_array,
                "example_ids_sha256": sha256_json(example_ids),
                "source_indices_sha256": sha256_json(source_indices),
                "source_cache_sha256": source_cache_sha256,
                "source_data_hash": source_data_hash,
                "raw_target_confusion_matrix_pre": target_pre_metrics[
                    "confusion_matrix"
                ],
                "raw_target_confusion_matrix_post": target_post_metrics[
                    "confusion_matrix"
                ],
                "raw_control_confusion_matrix_pre": control_pre_metrics[
                    "confusion_matrix"
                ],
                "raw_control_confusion_matrix_post": control_post_metrics[
                    "confusion_matrix"
                ],
            }
        )


def _write_matched_rows(
    context: RunContext,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    for row in rows:
        key = (
            row["model_key"],
            row["task"],
            int(row["layer"]),
            int(row["pair_seed"]),
            row["method"],
            row["condition"],
        )
        context.write_json_shard("matched_split", key, dict(row))


def _materialize_shards(
    context: RunContext,
    *,
    experiment: str,
    expected_keys: Iterable[Any],
    csv_name: str,
    parquet_name: str,
    key_columns: Iterable[str],
) -> pd.DataFrame:
    keys = tuple(expected_keys)
    observed = context.observed_keys(experiment, expected_keys=keys)
    expected_tuples = {tuple(key) for key in keys}
    if observed != expected_tuples:
        missing = sorted(expected_tuples - observed)
        raise RuntimeError(
            f"{experiment} is incomplete: {len(missing)} exact keys missing; first={missing[:3]}"
        )
    rows = [context.load_json_shard(experiment, key) for key in keys]
    return materialize_rows(
        rows,
        csv_path=context.run_dir / csv_name,
        parquet_path=context.run_dir / parquet_name,
        key_columns=tuple(key_columns),
    )


def run_matched_split(
    context: RunContext,
    config: RevisionConfig,
    *,
    device: torch.device,
) -> dict[str, Any]:
    if device.type != "cpu":
        LOGGER.info(
            "matched/split scientific probes and edits are pinned to CPU; "
            "ignoring requested accelerator %s",
            device,
        )
    device = torch.device("cpu")
    _require_ok_report(context, "preflight_report.json")
    _require_ok_report(context, "baseline_reproduction.json")
    benchmark = _load_benchmark_report(context)
    grid = benchmark["selected_grid"]
    expected_keys = config.matched_split_row_keys(grid)
    completed = context.completed_keys("matched_split", expected_keys=expected_keys)
    cache_pins = _preflight_cache_pins(context)
    reconstructions = _reconstruct_tasks(config)
    split_seed = int(config.raw["reproducibility"]["master_seed"])
    fatal_errors: list[str] = []

    for model in config.models:
        selected_layers = [
            cell.layer
            for cell in config.matched_split_cells(grid)
            if cell.model_key == model["key"] and cell.task == config.tasks[0]
        ]
        for task in config.tasks:
            reconstruction = reconstructions[task]
            subdivision, indices = _three_way_scoring_split(
                reconstruction, seed=split_seed
            )
            split_manifest_ref, split_manifest_sha256 = _persist_scoring_split_manifest(
                context,
                reconstruction=reconstruction,
                subdivision=subdivision,
                indices=indices,
                seed=split_seed,
            )
            for layer in selected_layers:
                cell_prefixes = {
                    (model["key"], task, layer, seed)
                    for seed in config.pair_seeds
                }
                if all(
                    any(key[:4] == prefix for key in completed)
                    and all(
                        key in completed
                        for key in expected_keys
                        if tuple(key[:4]) == prefix
                    )
                    for prefix in cell_prefixes
                ):
                    continue
                LOGGER.info("matched/split: %s/%s/L%s", model["key"], task, layer)
                cache = _load_cache(
                    model_id=model["hf_id"],
                    task=task,
                    layer=layer,
                    tag="inter",
                    reconstruction=reconstruction,
                    expected_cache_sha256=cache_pins[
                        (model["hf_id"], task, int(layer), "inter")
                    ],
                )
                X_score = cache.X[indices["score"]]
                zc_score = cache.zc[indices["score"]]
                ze_score = cache.ze[indices["score"]]
                score_manifest_rows = _subset_manifest_rows(subdivision, "score")
                score_example_ids = [row["example_id"] for row in score_manifest_rows]
                score_source_indices = [int(value) for value in indices["score"].tolist()]

                reference_edits: dict[str, torch.Tensor | Exception] = {}
                for method in ("inlp", "rlace"):
                    try:
                        set_seed(split_seed + layer)
                        reference_edits[method] = reference_edit(
                            method,
                            X_score,
                            zc_score,
                            device=device,
                            inlp_iterations=10,
                            rlace_rank=1,
                            rlace_steps=500,
                        )
                    except Exception as error:  # noqa: BLE001 - persist explicit unit failure
                        reference_edits[method] = error

                for pair_seed in config.pair_seeds:
                    try:
                        pair = _load_or_train_pair(
                            context,
                            model_key=model["key"],
                            task=task,
                            layer=layer,
                            pair_seed=pair_seed,
                            cache=cache,
                            indices=indices,
                            split_hashes=subdivision.subset_hashes,
                            split_manifest_ref=split_manifest_ref,
                            split_manifest_sha256=split_manifest_sha256,
                            device=device,
                        )
                    except Exception as error:  # noqa: BLE001 - persist exact failures
                        fatal_errors.append(
                            f"{model['key']}/{task}/L{layer}/{pair_seed}/probe_training: {error}"
                        )
                        for method in ("alterrep", "fgsm", "pgd", "inlp", "rlace"):
                            conditions = (
                                ("reference",)
                                if method in {"inlp", "rlace"}
                                else ("matched", "split")
                            )
                            common = _base_row(
                                context,
                                config,
                                model=model,
                                task=task,
                                layer=layer,
                                pair_seed=pair_seed,
                                method=method,
                                cache=cache,
                                subdivision=subdivision,
                                indices=indices,
                                pair=None,
                                split_manifest_ref=split_manifest_ref,
                                split_manifest_sha256=split_manifest_sha256,
                            )
                            rows = _failure_rows(
                                common,
                                conditions=conditions,
                                error=error,
                                failure_stage="probe_training",
                            )
                            _write_matched_rows(context, rows)
                            for condition in conditions:
                                completed.add(
                                    (
                                        model["key"],
                                        task,
                                        layer,
                                        pair_seed,
                                        method,
                                        condition,
                                    )
                                )
                        continue
                    pre_metrics = _pre_edit_metrics(
                        pair,
                        X_score,
                        zc_score,
                        ze_score,
                        device=device,
                    )
                    for method in ("alterrep", "fgsm", "pgd"):
                        keys = [
                            (model["key"], task, layer, pair_seed, method, condition)
                            for condition in ("matched", "split")
                        ]
                        if all(key in completed for key in keys):
                            continue
                        common = _base_row(
                            context,
                            config,
                            model=model,
                            task=task,
                            layer=layer,
                            pair_seed=pair_seed,
                            method=method,
                            cache=cache,
                            subdivision=subdivision,
                            indices=indices,
                            pair=pair,
                            split_manifest_ref=split_manifest_ref,
                            split_manifest_sha256=split_manifest_sha256,
                        )
                        started = time.perf_counter()
                        try:
                            if method == "alterrep":
                                edited = alterrep_edit(
                                    X_score,
                                    zc_score,
                                    pair.attacker.target_probe,
                                    device=device,
                                    alpha=1.0,
                                )
                                linf = None
                            else:
                                edited = attack_edit(
                                    method,
                                    X_score,
                                    zc_score,
                                    pair.attacker.target_probe,
                                    device=device,
                                    epsilon=0.5,
                                    pgd_steps=10,
                                )
                                linf = realized_linf_norm(X_score, edited)
                            rows = score_conditions_for_edit(
                                X_pre=X_score,
                                X_post=edited,
                                target_labels=zc_score,
                                control_labels=ze_score,
                                matched_target_probe=pair.attacker.target_probe,
                                matched_control_probe=pair.attacker.control_probe,
                                split_target_probe=pair.evaluator.target_probe,
                                split_control_probe=pair.evaluator.control_probe,
                                device=device,
                                common=common,
                            )
                            elapsed = time.perf_counter() - started
                            for row in rows:
                                _normalize_score_status(row)
                                row["realized_linf_norm"] = linf
                                row["wall_seconds"] = elapsed
                            _augment_matched_artifacts(
                                context,
                                rows,
                                pair=pair,
                                X_pre=X_score,
                                X_post=edited,
                                target_labels=zc_score,
                                control_labels=ze_score,
                                example_ids=score_example_ids,
                                source_indices=score_source_indices,
                                split_name="score",
                                source_cache_sha256=cache.selection.cache_sha256,
                                source_data_hash=cache.selection.data_hash,
                                method=method,
                                device=device,
                            )
                        except Exception as error:  # noqa: BLE001 - persist explicit unit failure
                            rows = _failure_rows(
                                common,
                                conditions=("matched", "split"),
                                error=error,
                                failure_stage="edit_generation_or_scoring",
                                pre_metrics=pre_metrics,
                            )
                            fatal_errors.append(f"{model['key']}/{task}/L{layer}/{pair_seed}/{method}: {error}")
                        _write_matched_rows(context, rows)
                        completed.update(keys)

                    for method in ("inlp", "rlace"):
                        key = (model["key"], task, layer, pair_seed, method, "reference")
                        if key in completed:
                            continue
                        common = _base_row(
                            context,
                            config,
                            model=model,
                            task=task,
                            layer=layer,
                            pair_seed=pair_seed,
                            method=method,
                            cache=cache,
                            subdivision=subdivision,
                            indices=indices,
                            pair=pair,
                            split_manifest_ref=split_manifest_ref,
                            split_manifest_sha256=split_manifest_sha256,
                        )
                        reference = reference_edits[method]
                        if isinstance(reference, Exception):
                            rows = _failure_rows(
                                common,
                                conditions=("reference",),
                                error=reference,
                                failure_stage="reference_edit_generation",
                                pre_metrics=pre_metrics,
                            )
                            fatal_errors.append(f"{model['key']}/{task}/L{layer}/{method}: {reference}")
                        else:
                            scored = score_conditions_for_edit(
                                X_pre=X_score,
                                X_post=reference,
                                target_labels=zc_score,
                                control_labels=ze_score,
                                matched_target_probe=pair.evaluator.target_probe,
                                matched_control_probe=pair.evaluator.control_probe,
                                split_target_probe=pair.evaluator.target_probe,
                                split_control_probe=pair.evaluator.control_probe,
                                device=device,
                                common=common,
                            )[0]
                            scored["condition"] = "reference"
                            _normalize_score_status(scored)
                            scored["realized_linf_norm"] = None
                            scored["wall_seconds"] = None
                            rows = [scored]
                            _augment_matched_artifacts(
                                context,
                                rows,
                                pair=pair,
                                X_pre=X_score,
                                X_post=reference,
                                target_labels=zc_score,
                                control_labels=ze_score,
                                example_ids=score_example_ids,
                                source_indices=score_source_indices,
                                split_name="score",
                                source_cache_sha256=cache.selection.cache_sha256,
                                source_data_hash=cache.selection.data_hash,
                                method=method,
                                device=device,
                            )
                        _write_matched_rows(context, rows)
                        completed.add(key)
                del cache
                gc.collect()

    frame = _materialize_shards(
        context,
        experiment="matched_split",
        expected_keys=expected_keys,
        csv_name="matched_split_rows.csv",
        parquet_name="matched_split_rows.parquet",
        key_columns=("model_key", "task", "layer", "pair_seed", "method", "condition"),
    )
    if fatal_errors or _hard_failure_mask(frame).any():
        context.update_manifest(
            {"matched_split": {"status": "failed", "errors": fatal_errors[:20]}}
        )
        raise RuntimeError(f"matched/split contains {len(fatal_errors)} failed units")
    summary = summarize_matched_split(
        frame,
        expected_keys=expected_keys,
        key_columns=("model_key", "task", "layer", "pair_seed", "method", "condition"),
        expected_blocks=12,
        bootstrap_draws=int(config.raw["reproducibility"]["bootstrap_draws"]),
        bootstrap_seed=int(config.raw["reproducibility"]["bootstrap_seed"]),
        permutation_seed=int(config.raw["reproducibility"]["permutation_seed"]),
        raw_row_file_sha256=sha256_file(context.run_dir / "matched_split_rows.parquet"),
        generating_git_commit=_current_commit(),
    )
    summary["status"] = "ok"
    summary["selected_grid"] = grid
    atomic_write_json(context.run_dir / "matched_split_summary.json", summary)
    context.update_manifest(
        {"matched_split": {"status": "ok", "grid": grid, "rows": len(frame)}}
    )
    return summary


def _epsilon_expected_keys(
    config: RevisionConfig,
    *,
    scope: str,
    grid: str,
) -> tuple[tuple[Any, ...], ...]:
    if scope == "middle_only":
        return tuple(tuple(key) for key in config.epsilon_sweep_row_keys())
    cells = config.matched_split_cells(grid)
    return tuple(
        (
            cell.model_key,
            cell.task,
            cell.layer,
            pair_seed,
            method,
            float(epsilon),
            condition,
        )
        for cell in cells
        for pair_seed in config.pair_seeds
        for method in ("fgsm", "pgd")
        for epsilon in config.epsilons
        for condition in ("matched", "split")
    )


def _epsilon_array_path(context: RunContext, key: tuple[Any, ...]) -> Path:
    return context.run_dir / "arrays" / "epsilon" / f"{sha256_json(list(key))}.pt"


def _epsilon_edit_path(context: RunContext, key: tuple[Any, ...]) -> Path:
    return (
        context.run_dir
        / "arrays"
        / "epsilon"
        / "edits"
        / f"{sha256_json(list(key))}.pt"
    )


def _write_epsilon_rows(context: RunContext, rows: Iterable[Mapping[str, Any]]) -> None:
    for row in rows:
        key = (
            row["model_key"],
            row["task"],
            int(row["layer"]),
            int(row["pair_seed"]),
            row["method"],
            float(row["epsilon"]),
            row["condition"],
        )
        context.write_json_shard("epsilon_sweep", key, dict(row))


def _augment_epsilon_metrics(
    context: RunContext,
    rows: list[dict[str, Any]],
    *,
    pair: AttackerEvaluatorPair,
    X_pre: torch.Tensor,
    X_post: torch.Tensor,
    target_labels: torch.Tensor,
    control_labels: torch.Tensor,
    example_ids: list[str],
    split_name: str,
    source_cache_sha256: str,
    source_data_hash: str,
    source_indices: list[int] | None,
    epsilon: float,
    method: str,
    device: torch.device,
) -> None:
    if len(example_ids) != len(X_pre):
        raise ValueError("epsilon example IDs are not aligned to representations")
    if X_pre.shape != X_post.shape:
        raise ValueError("epsilon pre/post representation shapes differ")
    if source_indices is not None and len(source_indices) != len(X_pre):
        raise ValueError("epsilon source indices are not aligned to representations")
    edit_key = (
        rows[0]["model_key"],
        rows[0]["task"],
        rows[0]["layer"],
        rows[0]["pair_seed"],
        method,
        float(epsilon),
    )
    delta = (X_post.detach().cpu().float() - X_pre.detach().cpu().float()).contiguous()
    edit_hash = sha256_tensor(X_post.detach().cpu().float().contiguous())
    edit_path = _epsilon_edit_path(context, edit_key)
    _write_immutable_torch(
        edit_path,
        {
            "schema_version": 2,
            "key": list(edit_key),
            "example_ids": list(example_ids),
            "source_indices": list(source_indices) if source_indices is not None else None,
            "split_names": [split_name] * len(example_ids),
            "source_cache_sha256": source_cache_sha256,
            "source_data_hash": source_data_hash,
            "source_representation_dtype": str(X_pre.dtype),
            "representation_delta": delta,
            "reconstruction_rule": "edited_representations = source_cache_rows + representation_delta",
            "edit_hash": edit_hash,
            "delta_sha256": sha256_tensor(delta),
        },
    )
    relative_edit = _relative(edit_path, context.run_dir)
    probes = {
        "matched": (pair.attacker.target_probe, pair.attacker.control_probe),
        "split": (pair.evaluator.target_probe, pair.evaluator.control_probe),
    }
    for row in rows:
        condition = row["condition"]
        target_probe, control_probe = probes[condition]
        target_logits_pre = evaluate_logits(target_probe, X_pre, device)
        target_logits_post = evaluate_logits(target_probe, X_post, device)
        control_logits_pre = evaluate_logits(control_probe, X_pre, device)
        control_logits_post = evaluate_logits(control_probe, X_post, device)
        target_probabilities_pre = torch.softmax(target_logits_pre.float(), dim=1)
        target_probabilities_post = torch.softmax(target_logits_post.float(), dim=1)
        control_probabilities_pre = torch.softmax(control_logits_pre.float(), dim=1)
        control_probabilities_post = torch.softmax(control_logits_post.float(), dim=1)
        target_pre_metrics = binary_metrics_from_logits(target_logits_pre, target_labels)
        target_metrics = binary_metrics_from_logits(target_logits_post, target_labels)
        control_pre_metrics = binary_metrics_from_logits(control_logits_pre, control_labels)
        control_post_metrics = binary_metrics_from_logits(control_logits_post, control_labels)
        key = (
            row["model_key"], row["task"], row["layer"], row["pair_seed"],
            method, float(epsilon), condition,
        )
        array_path = _epsilon_array_path(context, key)
        checkpoint_hashes = (
            {
                "target": pair.attacker.target_checkpoint_hash,
                "control": pair.attacker.control_checkpoint_hash,
            }
            if condition == "matched"
            else {
                "target": pair.evaluator.target_checkpoint_hash,
                "control": pair.evaluator.control_checkpoint_hash,
            }
        )
        _write_immutable_torch(
            array_path,
            {
                "schema_version": 2,
                "key": list(key),
                "example_ids": list(example_ids),
                "source_indices": (
                    list(source_indices) if source_indices is not None else None
                ),
                "split_names": [split_name] * len(example_ids),
                "source_cache_sha256": source_cache_sha256,
                "source_data_hash": source_data_hash,
                "edit_artifact_ref": relative_edit,
                "edit_hash": edit_hash,
                "evaluator_checkpoint_hashes": checkpoint_hashes,
                "target_logits_pre": target_logits_pre,
                "target_logits_post": target_logits_post,
                "target_probabilities_pre": target_probabilities_pre,
                "target_probabilities_post": target_probabilities_post,
                "target_predictions_pre": target_logits_pre.argmax(dim=1),
                "target_predictions_post": target_logits_post.argmax(dim=1),
                "target_labels": target_labels.detach().cpu(),
                "control_logits_pre": control_logits_pre,
                "control_logits_post": control_logits_post,
                "control_probabilities_pre": control_probabilities_pre,
                "control_probabilities_post": control_probabilities_post,
                "control_predictions_pre": control_logits_pre.argmax(dim=1),
                "control_predictions_post": control_logits_post.argmax(dim=1),
                "control_labels": control_labels.detach().cpu(),
                "confusion_matrices": {
                    "target_pre": target_pre_metrics["confusion_matrix"],
                    "target_post": target_metrics["confusion_matrix"],
                    "control_pre": control_pre_metrics["confusion_matrix"],
                    "control_post": control_post_metrics["confusion_matrix"],
                },
            },
        )
        relative_array = _relative(array_path, context.run_dir)
        row.update(
            {
                "epsilon": float(epsilon),
                "pgd_steps": 10 if method == "pgd" else None,
                "pgd_step_size": pgd_step_size(epsilon) if method == "pgd" else None,
                "random_start_seed": None,
                "random_start_behavior": "none_in_canonical_baseline",
                "edit_artifact_ref": relative_edit,
                "per_example_artifact_ref": relative_array,
                "epsilon_edit_hash": edit_hash,
                "epsilon_delta_sha256": sha256_tensor(delta),
                "example_ids_sha256": sha256_json(example_ids),
                "source_indices_sha256": (
                    sha256_json(source_indices) if source_indices is not None else None
                ),
                "source_cache_sha256": source_cache_sha256,
                "source_data_hash": source_data_hash,
                "raw_target_logits_pre_path_or_ref": relative_array + "#target_logits_pre",
                "raw_target_logits_post_path_or_ref": relative_array + "#target_logits_post",
                "raw_target_predictions_pre_path_or_ref": relative_array + "#target_predictions_pre",
                "raw_target_predictions_post_path_or_ref": relative_array + "#target_predictions_post",
                "raw_target_accuracy_pre": row["target_acc_pre"],
                "raw_target_accuracy_post": row["target_acc_post"],
                "raw_control_accuracy_pre": row["control_acc_pre"],
                "raw_control_accuracy_post": row["control_acc_post"],
                "orientation_sensitivity_accuracy": target_metrics[
                    "complement_sensitivity_accuracy"
                ],
                "orientation_sensitivity_accuracy_post": target_metrics[
                    "complement_sensitivity_accuracy"
                ],
                "auc": target_metrics["auc"],
                "auc_post": target_metrics["auc"],
                "log_loss": target_metrics["log_loss"],
                "log_loss_post": target_metrics["log_loss"],
                "raw_target_confusion_matrix_pre": target_pre_metrics[
                    "confusion_matrix"
                ],
                "raw_target_confusion_matrix_post": target_metrics[
                    "confusion_matrix"
                ],
                "raw_control_confusion_matrix_pre": control_pre_metrics[
                    "confusion_matrix"
                ],
                "raw_control_confusion_matrix_post": control_post_metrics[
                    "confusion_matrix"
                ],
                "ceiling_indicator": bool(
                    row.get("C") is not None and abs(float(row["C"]) - 1.0) <= 1.0e-12
                ),
            }
        )


def _validate_epsilon_baseline_against_matched(
    matched_frame: pd.DataFrame,
    epsilon_frame: pd.DataFrame,
    *,
    expected_rows: int = 240,
) -> dict[str, Any]:
    """Require epsilon=.5 to reproduce Experiment A row for row."""

    key_columns = [
        "model_key",
        "task",
        "layer",
        "pair_seed",
        "method",
        "condition",
    ]
    numeric_columns = [
        "C",
        "S",
        "H",
        "target_acc_pre",
        "target_acc_post",
        "control_acc_pre",
        "control_acc_post",
    ]
    comparison_columns = [*key_columns, "edit_hash", "status", *numeric_columns]
    for name, comparison_frame in (
        ("Experiment A", matched_frame),
        ("epsilon sweep", epsilon_frame),
    ):
        missing = [
            column for column in comparison_columns if column not in comparison_frame
        ]
        if missing:
            raise ValueError(f"{name} comparison rows lack columns: {missing!r}")
    required = epsilon_frame.loc[
        (epsilon_frame["epsilon_scope"] == "required_middle")
        & np.isclose(epsilon_frame["epsilon"].astype(float), 0.5)
    ].copy()
    matched_candidates = matched_frame.loc[
        matched_frame["method"].isin(("fgsm", "pgd"))
        & matched_frame["condition"].isin(("matched", "split"))
    ].copy()
    required_key_frame = required[key_columns].drop_duplicates()
    matched = matched_candidates.merge(
        required_key_frame,
        on=key_columns,
        how="inner",
        validate="many_to_one",
    )
    if len(required) != expected_rows:
        raise ValueError(
            f"epsilon=.5 comparison has {len(required)} rows, expected {expected_rows}"
        )
    if required.duplicated(key_columns).any() or matched.duplicated(key_columns).any():
        raise ValueError("epsilon=.5 comparison contains duplicate scientific keys")
    required_keys = set(map(tuple, required[key_columns].itertuples(index=False, name=None)))
    matched_keys = set(map(tuple, matched[key_columns].itertuples(index=False, name=None)))
    if required_keys != matched_keys:
        missing = sorted(matched_keys - required_keys)
        unexpected = sorted(required_keys - matched_keys)
        raise ValueError(
            f"epsilon=.5 keys differ from Experiment A: missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )
    combined = required.merge(
        matched,
        on=key_columns,
        how="inner",
        validate="one_to_one",
        suffixes=("_epsilon", "_matched"),
    )
    hash_mismatch = combined["edit_hash_epsilon"].astype(str) != combined[
        "edit_hash_matched"
    ].astype(str)
    if hash_mismatch.any():
        raise ValueError("epsilon=.5 edit hash differs from Experiment A")
    status_mismatch = combined["status_epsilon"].astype(str) != combined[
        "status_matched"
    ].astype(str)
    if status_mismatch.any():
        raise ValueError("epsilon=.5 score status differs from Experiment A")
    maximum_deviations: dict[str, float] = {}
    for column in numeric_columns:
        left = combined[f"{column}_epsilon"].astype(float).to_numpy()
        right = combined[f"{column}_matched"].astype(float).to_numpy()
        asymmetric_null = np.isnan(left) != np.isnan(right)
        if asymmetric_null.any():
            raise ValueError(
                f"epsilon=.5 {column} null pattern differs from Experiment A"
            )
        finite = np.isfinite(left) & np.isfinite(right)
        deviation = (
            float(np.max(np.abs(left[finite] - right[finite])))
            if finite.any()
            else 0.0
        )
        maximum_deviations[column] = deviation
        if not np.allclose(left, right, rtol=0.0, atol=1.0e-12, equal_nan=True):
            raise ValueError(
                f"epsilon=.5 {column} differs from Experiment A; max={deviation}"
            )
    return {
        "passed": True,
        "rows": len(combined),
        "edit_hashes_equal": True,
        "score_statuses_equal": True,
        "numeric_absolute_tolerance": 1.0e-12,
        "maximum_absolute_deviations": maximum_deviations,
    }


def _summarize_epsilon_half_pattern(
    required_middle_rows: pd.DataFrame,
    *,
    expected_rows: int = 240,
) -> dict[str, Any]:
    """Describe the reproduced epsilon=.5 pattern without assuming all rows ceil."""

    missing = [
        column
        for column in ("epsilon", "C", "status")
        if column not in required_middle_rows
    ]
    if missing:
        raise RuntimeError(f"epsilon=0.5 pattern rows lack columns: {missing!r}")
    baseline = required_middle_rows.loc[
        np.isclose(required_middle_rows["epsilon"].astype(float), 0.5)
    ].copy()
    if len(baseline) != expected_rows:
        raise RuntimeError(
            f"epsilon=0.5 required scope has {len(baseline)} rows; "
            f"expected {expected_rows}"
        )
    defined = pd.to_numeric(baseline["C"], errors="coerce").dropna().to_numpy(
        dtype=float
    )
    if defined.size and not np.isfinite(defined).all():
        raise RuntimeError("epsilon=0.5 required scope contains non-finite C values")
    ceiling_rows = int(np.isclose(defined, 1.0, atol=1.0e-12).sum())
    return {
        "rows": len(baseline),
        "defined_score_rows": int(defined.size),
        "score_null_rows": int(len(baseline) - defined.size),
        "ceiling_rows": ceiling_rows,
        "ceiling_fraction_among_defined": (
            float(ceiling_rows / defined.size) if defined.size else None
        ),
        "status_counts": {
            str(status): int(count)
            for status, count in baseline["status"]
            .astype(str)
            .value_counts()
            .sort_index()
            .items()
        },
        "interpretation": "descriptive_pattern_reproduced_exactly_from_experiment_a",
    }


def run_epsilon_sweep(
    context: RunContext,
    config: RevisionConfig,
    *,
    device: torch.device,
) -> dict[str, Any]:
    if device.type != "cpu":
        LOGGER.info(
            "epsilon scientific probes and attacks are pinned to CPU; "
            "ignoring requested accelerator %s",
            device,
        )
    device = torch.device("cpu")
    _require_ok_report(context, "preflight_report.json")
    benchmark = _load_benchmark_report(context)
    matched_summary = _require_ok_report(context, "matched_split_summary.json")
    del matched_summary
    grid = benchmark["selected_grid"]
    scope = benchmark["epsilon_scope"]
    expected_keys = _epsilon_expected_keys(config, scope=scope, grid=grid)
    completed = context.completed_keys("epsilon_sweep", expected_keys=expected_keys)
    cache_pins = _preflight_cache_pins(context)
    cells = (
        config.matched_split_cells(grid)
        if scope == "all_selected_layers"
        else tuple(
            cell
            for cell in config.matched_split_cells("full")
            if cell.layer == config.model(cell.model_key)["middle_layer"]
        )
    )
    reconstructions = _reconstruct_tasks(config)
    split_seed = int(config.raw["reproducibility"]["master_seed"])
    fatal_errors: list[str] = []

    for cell in cells:
        model = config.model(cell.model_key)
        reconstruction = reconstructions[cell.task]
        subdivision, indices = _three_way_scoring_split(
            reconstruction, seed=split_seed
        )
        split_manifest_ref, split_manifest_sha256 = _persist_scoring_split_manifest(
            context,
            reconstruction=reconstruction,
            subdivision=subdivision,
            indices=indices,
            seed=split_seed,
        )
        cache = _load_cache(
            model_id=model["hf_id"],
            task=cell.task,
            layer=cell.layer,
            tag="inter",
            reconstruction=reconstruction,
            expected_cache_sha256=cache_pins[
                (model["hf_id"], cell.task, int(cell.layer), "inter")
            ],
        )
        X_score = cache.X[indices["score"]]
        zc_score = cache.zc[indices["score"]]
        ze_score = cache.ze[indices["score"]]
        score_example_ids = [
            row["example_id"] for row in _subset_manifest_rows(subdivision, "score")
        ]
        for pair_seed in config.pair_seeds:
            pair, metadata = load_pair_checkpoint(
                _checkpoint_path(
                    context,
                    model_key=cell.model_key,
                    task=cell.task,
                    layer=cell.layer,
                    pair_seed=pair_seed,
                ),
                device=device,
            )
            if metadata.get("cache_sha256") != cache.selection.cache_sha256:
                raise RuntimeError("epsilon sweep checkpoint/cache provenance mismatch")
            pre_metrics = _pre_edit_metrics(
                pair,
                X_score,
                zc_score,
                ze_score,
                device=device,
            )
            for method in ("fgsm", "pgd"):
                for epsilon in config.epsilons:
                    keys = [
                        (
                            cell.model_key, cell.task, cell.layer, pair_seed,
                            method, float(epsilon), condition,
                        )
                        for condition in ("matched", "split")
                    ]
                    if all(key in completed for key in keys):
                        continue
                    common = _base_row(
                        context,
                        config,
                        model=model,
                        task=cell.task,
                        layer=cell.layer,
                        pair_seed=pair_seed,
                        method=method,
                        cache=cache,
                        subdivision=subdivision,
                        indices=indices,
                        pair=pair,
                        split_manifest_ref=split_manifest_ref,
                        split_manifest_sha256=split_manifest_sha256,
                    )
                    common["epsilon_scope"] = (
                        "required_middle"
                        if cell.layer == int(model["middle_layer"])
                        else "automatic_extension"
                    )
                    started = time.perf_counter()
                    try:
                        edited = attack_edit(
                            method,
                            X_score,
                            zc_score,
                            pair.attacker.target_probe,
                            device=device,
                            epsilon=float(epsilon),
                            pgd_steps=10,
                        )
                        realized = validate_linf_edit(
                            X_score, edited, float(epsilon), tolerance=1.0e-6
                        )
                        rows = score_conditions_for_edit(
                            X_pre=X_score,
                            X_post=edited,
                            target_labels=zc_score,
                            control_labels=ze_score,
                            matched_target_probe=pair.attacker.target_probe,
                            matched_control_probe=pair.attacker.control_probe,
                            split_target_probe=pair.evaluator.target_probe,
                            split_control_probe=pair.evaluator.control_probe,
                            device=device,
                            common=common,
                        )
                        for row in rows:
                            _normalize_score_status(row)
                            row["realized_linf_norm"] = realized
                        _augment_epsilon_metrics(
                            context,
                            rows,
                            pair=pair,
                            X_pre=X_score,
                            X_post=edited,
                            target_labels=zc_score,
                            control_labels=ze_score,
                            example_ids=score_example_ids,
                            split_name="score",
                            source_cache_sha256=cache.selection.cache_sha256,
                            source_data_hash=cache.selection.data_hash,
                            source_indices=[
                                int(value) for value in indices["score"].tolist()
                            ],
                            epsilon=float(epsilon),
                            method=method,
                            device=device,
                        )
                        elapsed = time.perf_counter() - started
                        for row in rows:
                            row["wall_seconds"] = elapsed
                            if epsilon == 0 and row["status"] == "ok" and abs(float(row["C"])) > 1.0e-6:
                                raise ValueError("epsilon-zero target damage exceeds tolerance")
                    except Exception as error:  # noqa: BLE001 - persist explicit unit failure
                        rows = _failure_rows(
                            {**common, "epsilon": float(epsilon)},
                            conditions=("matched", "split"),
                            error=error,
                            failure_stage="attack_generation_or_scoring",
                            pre_metrics=pre_metrics,
                        )
                        for row in rows:
                            row.update(
                                {
                                    "pgd_steps": 10 if method == "pgd" else None,
                                    "pgd_step_size": pgd_step_size(float(epsilon)) if method == "pgd" else None,
                                }
                            )
                        fatal_errors.append(
                            f"{cell.model_key}/{cell.task}/L{cell.layer}/{pair_seed}/{method}/{epsilon}: {error}"
                        )
                    _write_epsilon_rows(context, rows)
                    completed.update(keys)
        del cache
        gc.collect()

    frame = _materialize_shards(
        context,
        experiment="epsilon_sweep",
        expected_keys=expected_keys,
        csv_name="epsilon_sweep_rows.csv",
        parquet_name="epsilon_sweep_rows.parquet",
        key_columns=("model_key", "task", "layer", "pair_seed", "method", "epsilon", "condition"),
    )
    if fatal_errors or _hard_failure_mask(frame).any():
        context.update_manifest(
            {"epsilon_sweep": {"status": "failed", "errors": fatal_errors[:20]}}
        )
        raise RuntimeError(f"epsilon sweep contains {len(fatal_errors)} failed units")
    matched_frame = pd.read_parquet(context.run_dir / "matched_split_rows.parquet")
    baseline_comparison = _validate_epsilon_baseline_against_matched(
        matched_frame, frame, expected_rows=240
    )
    required = frame.loc[frame["epsilon_scope"] == "required_middle"]
    baseline_pattern = _summarize_epsilon_half_pattern(required)
    summary = summarize_epsilon_sweep(
        frame,
        expected_keys=expected_keys,
        key_columns=("model_key", "task", "layer", "pair_seed", "method", "epsilon", "condition"),
        raw_row_file_sha256=sha256_file(context.run_dir / "epsilon_sweep_rows.parquet"),
        generating_git_commit=_current_commit(),
    )
    summary["status"] = "ok"
    summary["scope"] = scope
    summary["required_middle_rows"] = len(required)
    summary["baseline_ceiling_reproduced"] = True
    summary["epsilon_half_ceiling_pattern"] = baseline_pattern
    summary["epsilon_half_experiment_a_reproduction"] = baseline_comparison
    atomic_write_json(context.run_dir / "epsilon_sweep_summary.json", summary)
    context.update_manifest(
        {"epsilon_sweep": {"status": "ok", "scope": scope, "rows": len(frame)}}
    )
    return summary


def _construct_cell_record(cell: ConstructCell) -> dict[str, str | int]:
    return {
        "model_key": cell.model_key,
        "model_id": cell.model_id,
        "task": cell.task,
        "layer": int(cell.layer),
    }


def construct_artifact_root(run_dir: Path, cell: ConstructCell) -> Path:
    """Return the collision-free artifact root for one construct cell."""

    return Path(run_dir) / "construct" / "cells" / cell.slug


def construct_row_identity(
    cell: ConstructCell,
    *,
    edit_kind: str,
    architecture: str | None,
    seed: int,
) -> dict[str, Any]:
    """Return the full cell/edit identity persisted on construct rows."""

    return {
        **_construct_cell_record(cell),
        "edit_kind": edit_kind,
        "architecture": architecture,
        "candidate_seed": int(seed),
    }


@dataclass(frozen=True)
class _ConstructLayout:
    artifact_root: Path
    checkpoint_root: Path
    array_root: Path
    rows_csv: Path
    rows_parquet: Path
    group_rows_csv: Path | None
    group_rows_parquet: Path | None
    summary_path: Path
    shard_experiment: str
    cell: ConstructCell | None
    legacy: bool

    def shard_key(self, edit_key: Iterable[Any]) -> tuple[Any, ...]:
        key = tuple(edit_key)
        if self.legacy:
            return key
        if self.cell is None:  # pragma: no cover - constructor enforces this.
            raise RuntimeError("generic construct layout is missing its cell")
        return (
            self.cell.model_key,
            self.cell.model_id,
            self.cell.task,
            int(self.cell.layer),
            *key,
        )

    @property
    def manifest_key(self) -> str:
        if self.legacy:
            return "construct_check"
        if self.cell is None:  # pragma: no cover - constructor enforces this.
            raise RuntimeError("generic construct layout is missing its cell")
        return f"construct_cell.{self.cell.slug}"

    @property
    def summary_cell(self) -> dict[str, str | int]:
        if self.cell is None:
            if self.legacy:
                return {
                    "model_key": _PILOT_CONSTRUCT_CELL.model_key,
                    "task": _PILOT_CONSTRUCT_CELL.task,
                    "layer": _PILOT_CONSTRUCT_CELL.layer,
                }
            raise RuntimeError("generic construct layout is missing its cell")
        record = _construct_cell_record(self.cell)
        if self.legacy:
            record.pop("model_id")
        return record


def _construct_layout(
    run_dir: Path,
    cell: ConstructCell | None,
    *,
    legacy: bool = False,
) -> _ConstructLayout:
    run_dir = Path(run_dir)
    if legacy:
        return _ConstructLayout(
            artifact_root=run_dir / "construct",
            checkpoint_root=run_dir / "checkpoints" / "construct",
            array_root=run_dir / "arrays" / "construct",
            rows_csv=run_dir / "construct_check_rows.csv",
            rows_parquet=run_dir / "construct_check_rows.parquet",
            group_rows_csv=None,
            group_rows_parquet=None,
            summary_path=run_dir / "construct_check_summary.json",
            shard_experiment="construct_edits",
            cell=cell,
            legacy=True,
        )
    if cell is None:
        raise ValueError("generic construct layout requires a cell")
    artifact_root = construct_artifact_root(run_dir, cell)
    return _ConstructLayout(
        artifact_root=artifact_root,
        checkpoint_root=(
            run_dir / "checkpoints" / "construct" / "cells" / cell.slug
        ),
        array_root=run_dir / "arrays" / "construct" / "cells" / cell.slug,
        rows_csv=artifact_root / "construct_check_rows.csv",
        rows_parquet=artifact_root / "construct_check_rows.parquet",
        group_rows_csv=artifact_root / "construct_group_rows.csv",
        group_rows_parquet=artifact_root / "construct_group_rows.parquet",
        summary_path=artifact_root / "construct_check_summary.json",
        shard_experiment=f"construct_edits.{cell.slug}",
        cell=cell,
        legacy=False,
    )


def _construct_payload(
    payload: Mapping[str, Any],
    *,
    cell: ConstructCell,
    legacy: bool,
) -> dict[str, Any]:
    output = dict(payload)
    if not legacy:
        output["cell"] = _construct_cell_record(cell)
    return output


def _validate_construct_cell(config: RevisionConfig, cell: ConstructCell) -> None:
    if cell.task not in config.tasks:
        raise ValueError(f"construct cell task is not locked by the base config: {cell.task}")
    model = config.model(cell.model_key)
    expected = ConstructCell(
        model_key=str(model["key"]),
        model_id=str(model["hf_id"]),
        task=cell.task,
        layer=int(model["middle_layer"]),
    )
    if cell != expected:
        raise ValueError(
            "construct cell must match a locked model/task middle-layer cell: "
            f"expected={expected}, observed={cell}"
        )


def _construct_model(architecture: str, input_dim: int) -> torch.nn.Module:
    if architecture == "linear":
        return LinearProbe(input_dim)
    if architecture == "mlp":
        return MLPProbe(input_dim, hidden_dim=256)
    if architecture == "mka":
        return MKAProbe(input_dim, hidden_dim=256, mka_lambda=0.5, knn_k=10)
    raise ValueError(f"unknown candidate architecture: {architecture}")


def _save_named_model(
    path: Path,
    model: torch.nn.Module,
    *,
    architecture: str,
    seed: int,
    metadata: Mapping[str, Any],
) -> str:
    state = _state_payload(model)
    checkpoint_hash = state_dict_sha256(state)
    atomic_torch_save(
        path,
        {
            "schema_version": 1,
            "architecture": architecture,
            "input_dim": next(model.parameters()).shape[-1],
            "seed": int(seed),
            "state_dict": state,
            "checkpoint_sha256": checkpoint_hash,
            "metadata": dict(metadata),
        },
    )
    return checkpoint_hash


def _load_named_model(
    path: Path,
    *,
    device: torch.device,
    expected_metadata: Mapping[str, Any],
) -> tuple[torch.nn.Module, str, int]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1 or payload.get("metadata") != dict(expected_metadata):
        raise RuntimeError(f"named checkpoint provenance mismatch: {path.name}")
    model = _construct_model(payload["architecture"], int(payload["input_dim"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    checkpoint_hash = state_dict_sha256(_state_payload(model))
    if checkpoint_hash != payload["checkpoint_sha256"]:
        raise RuntimeError(f"named checkpoint hash mismatch: {path.name}")
    return model, checkpoint_hash, int(payload["seed"])


def _construct_candidate(
    context: RunContext,
    *,
    architecture: str,
    seed: int,
    X_candidate: torch.Tensor,
    labels: torch.Tensor,
    cache_hash: str,
    device: torch.device,
    checkpoint_root: Path | None = None,
    cell_metadata: Mapping[str, Any] | None = None,
) -> tuple[torch.nn.Module, str]:
    root = checkpoint_root or context.run_dir / "checkpoints" / "construct"
    path = root / "candidates" / f"{architecture}-seed{seed}.pt"
    metadata = {
        "cache_sha256": cache_hash,
        "training_split": "phase2_candidate",
        "repository_seed": 1000 + seed,
    }
    if cell_metadata is not None:
        metadata["cell"] = dict(cell_metadata)
    if path.is_file():
        model, checkpoint_hash, _ = _load_named_model(
            path, device=device, expected_metadata=metadata
        )
        return model, checkpoint_hash
    set_seed(1000 + seed)
    model = _construct_model(architecture, X_candidate.shape[1]).to(device)
    train_probe(
        model,
        X_candidate,
        labels,
        ProbeTrainConfig(epochs=50, lr=1.0e-3, weight_decay=0.01, batch_size=256),
        device,
    )
    checkpoint_hash = _save_named_model(
        path,
        model,
        architecture=architecture,
        seed=seed,
        metadata=metadata,
    )
    return model, checkpoint_hash


def _construct_attackers(
    context: RunContext,
    *,
    X_candidate: torch.Tensor,
    zc_candidate: torch.Tensor,
    ze_candidate: torch.Tensor,
    cache_hash: str,
    pair_seeds: Iterable[int],
    device: torch.device,
    checkpoint_root: Path | None = None,
    cell_metadata: Mapping[str, Any] | None = None,
) -> list[ProbePair]:
    root = checkpoint_root or context.run_dir / "checkpoints" / "construct"
    pairs: list[ProbePair] = []
    for pair_seed in pair_seeds:
        path = root / "attackers" / f"pair{pair_seed}.pt"
        metadata = {
            "cache_sha256": cache_hash,
            "training_split": "phase2_candidate",
            "pair_seed": int(pair_seed),
        }
        if cell_metadata is not None:
            metadata["cell"] = dict(cell_metadata)
        if path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            target, target_hash = _probe_from_state(
                payload["target"], X_candidate.shape[1], device
            )
            control, control_hash = _probe_from_state(
                payload["control"], X_candidate.shape[1], device
            )
            if payload.get("metadata") != metadata or payload.get("hashes") != {
                "target": target_hash,
                "control": control_hash,
            }:
                raise RuntimeError("construct attacker checkpoint provenance mismatch")
            pairs.append(
                ProbePair(target, control, target_hash, control_hash, 10_000 + pair_seed)
            )
            continue
        pair = train_probe_pair(
            X_candidate,
            zc_candidate,
            ze_candidate,
            seed=10_000 + int(pair_seed),
            device=device,
            config=ProbeTrainConfig(),
        )
        atomic_torch_save(
            path,
            {
                "schema_version": 1,
                "metadata": metadata,
                "target": _state_payload(pair.target_probe),
                "control": _state_payload(pair.control_probe),
                "hashes": {
                    "target": pair.target_checkpoint_hash,
                    "control": pair.control_checkpoint_hash,
                },
            },
        )
        pairs.append(pair)
    return pairs


def _construct_evaluators(
    context: RunContext,
    *,
    X_evaluator: torch.Tensor,
    zc_evaluator: torch.Tensor,
    ze_evaluator: torch.Tensor,
    cache_hash: str,
    device: torch.device,
    checkpoint_root: Path | None = None,
    cell_metadata: Mapping[str, Any] | None = None,
):
    root = checkpoint_root or context.run_dir / "checkpoints" / "construct"
    path = root / "evaluators.pt"
    metadata = {
        "cache_sha256": cache_hash,
        "training_split": "phase2_evaluator",
        "n_evaluators": 5,
        "bag_fraction": 0.8,
    }
    if cell_metadata is not None:
        metadata["cell"] = dict(cell_metadata)
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("metadata") != metadata:
            raise RuntimeError("construct evaluator checkpoint provenance mismatch")
        output = []
        from src.metrics import ValidationProbes
        from src.ws5_repaired import Evaluator

        for saved in payload["evaluators"]:
            target, target_hash = _probe_from_state(saved["target"], X_evaluator.shape[1], device)
            control, control_hash = _probe_from_state(saved["control"], X_evaluator.shape[1], device)
            if saved["hashes"] != {"target": target_hash, "control": control_hash}:
                raise RuntimeError("construct evaluator checkpoint hash mismatch")
            output.append(
                Evaluator(
                    ValidationProbes(target, control, 0.0, 0.0),
                    int(saved["seed"]),
                    float(saved["acc_zc_oos"]),
                    float(saved["acc_ze_oos"]),
                )
            )
        return output

    evaluators = train_independent_evaluators(
        X_evaluator,
        zc_evaluator,
        ze_evaluator,
        ProbeTrainConfig(),
        device,
        n_evaluators=5,
        bag_frac=0.8,
        min_acc=0.60,
    )
    saved = []
    for evaluator in evaluators:
        target_state = _state_payload(evaluator.probes.zc_probe)
        control_state = _state_payload(evaluator.probes.ze_probe)
        saved.append(
            {
                "seed": evaluator.seed,
                "acc_zc_oos": evaluator.acc_zc_oos,
                "acc_ze_oos": evaluator.acc_ze_oos,
                "target": target_state,
                "control": control_state,
                "hashes": {
                    "target": state_dict_sha256(target_state),
                    "control": state_dict_sha256(control_state),
                },
            }
        )
    atomic_torch_save(
        path,
        {"schema_version": 1, "metadata": metadata, "evaluators": saved},
    )
    return evaluators


def _fresh_checkpoint(
    path: Path,
    *,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_validation: torch.Tensor,
    y_validation: torch.Tensor,
    X_final: torch.Tensor,
    y_final: torch.Tensor,
    final_example_ids: list[str],
    spec: DecoderSpec,
    seed: int,
    device: torch.device,
    metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], str, list[dict[str, Any]], int, Path]:
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("metadata") != dict(metadata) or payload.get("spec") != asdict(spec):
            raise RuntimeError(f"fresh decoder provenance mismatch: {path.name}")
        model = _construct_model(spec.architecture, X_train.shape[1]).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        checkpoint_hash = state_dict_sha256(_state_payload(model))
        if checkpoint_hash != payload["checkpoint_sha256"]:
            raise RuntimeError(f"fresh decoder checkpoint hash mismatch: {path.name}")
        history = list(payload["history"])
        best_epoch = int(payload["best_epoch"])
    else:
        result = train_deterministic_probe(
            X_train,
            y_train,
            X_validation,
            y_validation,
            spec=spec,
            seed=seed,
            device=device,
        )
        model = result.model
        checkpoint_hash = result.checkpoint_sha256
        history = result.history
        best_epoch = result.best_epoch
        atomic_torch_save(
            path,
            {
                "schema_version": 1,
                "metadata": dict(metadata),
                "spec": asdict(spec),
                "seed": seed,
                "state_dict": result.state_dict,
                "checkpoint_sha256": checkpoint_hash,
                "history": history,
                "best_epoch": best_epoch,
            },
        )
    logits = evaluate_logits(model, X_final, device)
    metrics = binary_metrics_from_logits(logits, y_final)
    if len(final_example_ids) != len(y_final):
        raise ValueError("fresh-decoder final example IDs are not aligned")
    per_example_path = path.with_suffix(".per-example.pt")
    _write_immutable_torch(
        per_example_path,
        {
            "schema_version": 1,
            "metadata": dict(metadata),
            "checkpoint_sha256": checkpoint_hash,
            "example_ids": list(final_example_ids),
            "split_names": ["final_test"] * len(final_example_ids),
            "logits": logits.detach().cpu().float(),
            "probabilities": torch.softmax(logits.detach().cpu().float(), dim=1),
            "predictions": logits.detach().cpu().argmax(dim=1),
            "labels": y_final.detach().cpu().long(),
        },
    )
    return metrics, checkpoint_hash, history, best_epoch, per_example_path


def _construct_edit_id(edit_kind: str, architecture: str, seed: int) -> str:
    return f"{edit_kind}-{architecture}-seed{seed}"


def _construct_expected_keys(config: RevisionConfig) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (_construct_edit_id(key.edit_kind, key.architecture, key.seed), family, label)
        for key in config.construct_edit_keys()
        for family in ("fixed", "fresh_linear", "fresh_mlp")
        for label in ("target", "control")
    )


def _ratio(post_accuracy: float, baseline_accuracy: float) -> float:
    denominator = baseline_accuracy - 0.5
    if abs(denominator) < 1.0e-12:
        raise ValueError("fresh-decoder baseline denominator is zero")
    return (post_accuracy - 0.5) / denominator


def _construct_row_base(
    *,
    context: RunContext,
    config: RevisionConfig,
    cell: ConstructCell = _PILOT_CONSTRUCT_CELL,
    edit_id: str,
    edit_kind: str,
    architecture: str,
    seed: int,
    evaluation_family: str,
    label: str,
    edit_hash: str,
    candidate_checkpoint_hash: str | None,
    evaluator_checkpoint_hashes: list[str],
    split_manifest_ref: str,
    artifact_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "run_id": context.run_id,
        "git_commit": _current_commit(),
        "config_hash": config.config_hash,
        **_construct_cell_record(cell),
        "edit_id": edit_id,
        "edit_object": edit_kind,
        "architecture": architecture,
        "candidate_seed": int(seed),
        "evaluation_family": evaluation_family,
        "label": label,
        "edit_hash": edit_hash,
        "candidate_checkpoint_hash": candidate_checkpoint_hash,
        "evaluator_checkpoint_hashes": evaluator_checkpoint_hashes,
        "split_manifest_ref": split_manifest_ref,
        "status": "ok",
        "failure_reason": None,
    }
    if artifact_provenance is not None:
        if artifact_provenance.get("split_manifest_ref") != split_manifest_ref:
            raise ValueError("construct row split manifest reference is inconsistent")
        row.update(dict(artifact_provenance))
    return row


def _unique_construct_row_value(frame: pd.DataFrame, column: str) -> Any:
    if column not in frame:
        raise RuntimeError(f"construct rows are missing provenance field: {column}")
    values: dict[str, Any] = {}
    for value in frame[column]:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            raise RuntimeError(f"construct row provenance field is null: {column}")
        values.setdefault(canonical_json(value), value)
    if len(values) != 1:
        raise RuntimeError(
            f"construct rows disagree on provenance field {column}: "
            f"{sorted(values)[:5]}"
        )
    return next(iter(values.values()))


def _load_hashed_construct_json(
    context: RunContext,
    frame: pd.DataFrame,
    *,
    label: str,
    reference_column: str,
    hash_column: str,
) -> tuple[str, str, dict[str, Any]]:
    reference = _unique_construct_row_value(frame, reference_column)
    expected_hash = _unique_construct_row_value(frame, hash_column)
    if not isinstance(reference, str) or not reference:
        raise RuntimeError(f"construct {label} reference is not a relative path")
    if not isinstance(expected_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_hash
    ):
        raise RuntimeError(f"construct {label} row hash is not SHA-256")
    path = _run_relative_path(context, reference)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"construct {label} artifact is missing: {reference}")
    observed_hash = sha256_file(path)
    if observed_hash != expected_hash:
        raise RuntimeError(
            f"construct {label} hash mismatch: expected {expected_hash}, "
            f"observed {observed_hash}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"construct {label} artifact is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"construct {label} artifact must contain a JSON object")
    return reference, expected_hash, payload


def _regenerate_construct_provenance(
    context: RunContext,
    construct_rows: pd.DataFrame,
) -> dict[str, Any]:
    """Rebuild construct summary provenance from rows and referenced bytes."""

    frame = construct_rows.copy().reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("construct rows are empty")
    split_ref, split_sha256, split_payload = _load_hashed_construct_json(
        context,
        frame,
        label="split manifest",
        reference_column="split_manifest_ref",
        hash_column="split_manifest_sha256",
    )
    baseline_ref, baseline_sha256, baseline_payload = _load_hashed_construct_json(
        context,
        frame,
        label="fresh decoder baseline",
        reference_column="fresh_decoder_baseline_ref",
        hash_column="fresh_decoder_baseline_sha256",
    )
    hyperparameter_ref, hyperparameter_sha256, _ = _load_hashed_construct_json(
        context,
        frame,
        label="hyperparameter selection",
        reference_column="hyperparameter_selection_ref",
        hash_column="hyperparameter_selection_sha256",
    )

    split_hashes = split_payload.get("intervention_subdivision_hashes")
    if not isinstance(split_hashes, dict) or not split_hashes:
        raise RuntimeError("construct split manifest has no subdivision hashes")
    row_split_hashes = _unique_construct_row_value(frame, "split_hashes")
    if canonical_json(row_split_hashes) != canonical_json(split_hashes):
        raise RuntimeError(
            "construct row split hashes do not match the independently hashed manifest"
        )
    group_disjointness = split_payload.get("group_disjointness")
    if group_disjointness is not True:
        raise RuntimeError("construct split manifest does not certify group disjointness")

    if "pre_edit_accuracy" in frame:
        fresh_rows = frame.loc[
            frame.get("evaluation_family", pd.Series(index=frame.index, dtype=object))
            .astype(str)
            .str.startswith("fresh")
        ]
        for row in fresh_rows.itertuples(index=False):
            family = str(row.evaluation_family)
            label_name = str(row.label)
            try:
                expected_accuracy = float(
                    baseline_payload[family][label_name]["metrics"]["accuracy"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"construct baseline artifact lacks {family}/{label_name} accuracy"
                ) from exc
            observed_accuracy = float(row.pre_edit_accuracy)
            if not math.isfinite(observed_accuracy) or not math.isclose(
                observed_accuracy,
                expected_accuracy,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise RuntimeError(
                    "fresh baseline accuracy mismatch for "
                    f"{family}/{label_name}: row={observed_accuracy}, "
                    f"artifact={expected_accuracy}"
                )

    return {
        "split_hashes": dict(split_hashes),
        "group_disjointness": True,
        "split_manifest_ref": split_ref,
        "split_manifest_sha256": split_sha256,
        "fresh_decoder_unedited_baselines": baseline_payload,
        "fresh_decoder_baseline_ref": baseline_ref,
        "fresh_decoder_baseline_sha256": baseline_sha256,
        "hyperparameter_selection_ref": hyperparameter_ref,
        "hyperparameter_selection_sha256": hyperparameter_sha256,
        "referenced_artifact_hashes": {
            split_ref: split_sha256,
            baseline_ref: baseline_sha256,
            hyperparameter_ref: hyperparameter_sha256,
        },
    }


_CONSTRUCT_GROUP_KEY_COLUMNS = (
    "model_key",
    "model_id",
    "task",
    "layer",
    "row_kind",
    "edit_id",
    "evaluation_family",
    "decoder_seed",
    "label",
    "group_id",
)


def _single_construct_artifact_value(value: Any, *, field: str) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 1:
        raise RuntimeError(f"construct {field} must contain exactly the seed-0 artifact")
    item = value[0]
    if not isinstance(item, str) or not item:
        raise RuntimeError(f"construct {field} contains an invalid artifact value")
    return item


def _load_construct_decoder_per_example(
    context: RunContext,
    *,
    reference: str,
    checkpoint_sha256: str,
    final_example_ids: list[str],
    cell: ConstructCell,
    legacy_layout: bool,
    edit_id: str,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    if re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256) is None:
        raise RuntimeError("construct decoder checkpoint hash is not SHA-256")
    path = _run_relative_path(context, reference)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"construct decoder per-example artifact is missing: {reference}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise RuntimeError("construct decoder per-example artifact has an invalid schema")
    if payload.get("checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError("construct decoder per-example checkpoint hash mismatch")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError("construct decoder per-example metadata is missing")
    expected_metadata = {
        "edit_id": edit_id,
        "family": "fresh_linear",
        "label": label,
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise RuntimeError("construct decoder per-example metadata mismatch")
    if legacy_layout:
        if "cell" in metadata:
            raise RuntimeError("legacy construct decoder metadata unexpectedly contains a cell")
    elif metadata.get("cell") != _construct_cell_record(cell):
        raise RuntimeError("construct decoder per-example cell metadata mismatch")
    example_ids = payload.get("example_ids")
    if not isinstance(example_ids, list) or example_ids != final_example_ids:
        raise RuntimeError("construct decoder per-example IDs are not final-test aligned")
    split_names = payload.get("split_names")
    if split_names != ["final_test"] * len(final_example_ids):
        raise RuntimeError("construct decoder per-example split names are invalid")
    predictions = payload.get("predictions")
    labels = payload.get("labels")
    if not isinstance(predictions, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("construct decoder per-example predictions/labels are missing")
    predictions = predictions.detach().cpu()
    labels = labels.detach().cpu()
    if (
        predictions.ndim != 1
        or labels.ndim != 1
        or predictions.shape != labels.shape
        or len(predictions) != len(final_example_ids)
    ):
        raise RuntimeError("construct decoder per-example prediction shape mismatch")
    if (
        predictions.dtype == torch.bool
        or labels.dtype == torch.bool
        or torch.is_floating_point(predictions)
        or torch.is_floating_point(labels)
        or not torch.isin(predictions, torch.tensor([0, 1])).all()
        or not torch.isin(labels, torch.tensor([0, 1])).all()
    ):
        raise RuntimeError("construct decoder per-example predictions/labels are not binary integers")
    return predictions.long().eq(labels.long()), labels.long(), sha256_file(path)


def _construct_group_records(
    *,
    common: Mapping[str, Any],
    correct: torch.Tensor,
    final_group_ids: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for group_id in sorted(set(final_group_ids)):
        indices = [
            index for index, observed_group in enumerate(final_group_ids)
            if observed_group == group_id
        ]
        records.append(
            {
                **dict(common),
                "group_id": group_id,
                "n_examples": len(indices),
                "correct_count": int(correct[indices].sum().item()),
            }
        )
    return records


def _validate_construct_per_example_accuracy(
    correct: torch.Tensor,
    expected: Any,
    *,
    identity: str,
) -> None:
    try:
        expected_accuracy = float(expected)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"construct summarized accuracy is invalid for {identity}"
        ) from exc
    observed_accuracy = float(correct.double().mean().item())
    if not math.isfinite(expected_accuracy) or not math.isclose(
        observed_accuracy,
        expected_accuracy,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError(
            "construct per-example accuracy mismatch for "
            f"{identity}: rows={observed_accuracy}, summary={expected_accuracy}"
        )


def build_construct_group_rows(
    context: RunContext,
    config: RevisionConfig,
    *,
    cell: ConstructCell,
    legacy_layout: bool = False,
) -> pd.DataFrame:
    """Reconstruct fresh-linear seed-0 group sufficient statistics from artifacts."""

    _validate_construct_cell(config, cell)
    if legacy_layout and cell != _PILOT_CONSTRUCT_CELL:
        raise ValueError("the legacy construct layout is restricted to the pilot cell")
    layout = _construct_layout(context.run_dir, cell, legacy=legacy_layout)
    if not layout.rows_parquet.is_file():
        raise RuntimeError("construct rows must be materialized before group statistics")
    source = pd.read_parquet(layout.rows_parquet).reset_index(drop=True)
    if source.empty or (source["status"] != "ok").any():
        raise RuntimeError("construct group statistics require an all-ok construct cell")
    observed_cells = set(
        source[["model_key", "model_id", "task", "layer"]].itertuples(
            index=False, name=None
        )
    )
    expected_cell = {(cell.model_key, cell.model_id, cell.task, int(cell.layer))}
    if observed_cells != expected_cell:
        raise RuntimeError("construct source rows do not match the requested cell")

    split_ref, split_sha256, split_payload = _load_hashed_construct_json(
        context,
        source,
        label="split manifest",
        reference_column="split_manifest_ref",
        hash_column="split_manifest_sha256",
    )
    _, _, baseline_payload = _load_hashed_construct_json(
        context,
        source,
        label="fresh decoder baseline",
        reference_column="fresh_decoder_baseline_ref",
        hash_column="fresh_decoder_baseline_sha256",
    )
    if not legacy_layout:
        expected_record = _construct_cell_record(cell)
        if split_payload.get("cell") != expected_record:
            raise RuntimeError("construct split manifest cell identity mismatch")
        if baseline_payload.get("cell") != expected_record:
            raise RuntimeError("construct baseline artifact cell identity mismatch")
    subdivision = split_payload.get("intervention_subdivision")
    if not isinstance(subdivision, list):
        raise TypeError("construct split manifest lacks an intervention subdivision")
    final_rows = sorted(
        (
            row for row in subdivision
            if isinstance(row, Mapping) and row.get("subset") == "final_test"
        ),
        key=lambda row: row.get("position_in_subset", -1),
    )
    if not final_rows:
        raise RuntimeError("construct split manifest has no final-test rows")
    positions = [row.get("position_in_subset") for row in final_rows]
    if positions != list(range(len(final_rows))):
        raise RuntimeError("construct final-test manifest positions are not contiguous")
    final_example_ids = [row.get("example_id") for row in final_rows]
    final_group_ids = [row.get("group_id") for row in final_rows]
    if (
        any(not isinstance(value, str) or not value for value in final_example_ids)
        or len(set(final_example_ids)) != len(final_example_ids)
        or any(not isinstance(value, str) or not value for value in final_group_ids)
    ):
        raise RuntimeError("construct final-test manifest identities are invalid")

    provenance = {
        "run_id": _unique_construct_row_value(source, "run_id"),
        "git_commit": _unique_construct_row_value(source, "git_commit"),
        "config_hash": _unique_construct_row_value(source, "config_hash"),
        **_construct_cell_record(cell),
        "evaluation_family": "fresh_linear",
        "decoder_seed": 0,
        "split_manifest_ref": split_ref,
        "split_manifest_sha256": split_sha256,
        "status": "ok",
        "failure_stage": None,
        "failure_reason": None,
    }
    records: list[dict[str, Any]] = []
    baseline_labels: dict[str, torch.Tensor] = {}
    try:
        fresh_linear_baseline = baseline_payload["fresh_linear"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("construct baseline artifact lacks fresh_linear records") from exc
    for label in ("target", "control"):
        try:
            baseline = fresh_linear_baseline[label]
            reference = _single_construct_artifact_value(
                baseline["per_example_refs"], field=f"baseline {label} per-example refs"
            )
            checkpoint_sha256 = _single_construct_artifact_value(
                baseline["checkpoint_hashes"], field=f"baseline {label} checkpoint hashes"
            )
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"construct baseline artifact lacks fresh_linear/{label} provenance"
            ) from exc
        correct, observed_labels, per_example_sha256 = (
            _load_construct_decoder_per_example(
                context,
                reference=reference,
                checkpoint_sha256=checkpoint_sha256,
                final_example_ids=final_example_ids,
                cell=cell,
                legacy_layout=legacy_layout,
                edit_id="unedited",
                label=label,
            )
        )
        try:
            baseline_accuracy = baseline["metrics"]["accuracy"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"construct baseline artifact lacks fresh_linear/{label} accuracy"
            ) from exc
        _validate_construct_per_example_accuracy(
            correct,
            baseline_accuracy,
            identity=f"baseline/{label}",
        )
        baseline_labels[label] = observed_labels
        records.extend(
            _construct_group_records(
                common={
                    **provenance,
                    "row_kind": "baseline",
                    "label": label,
                    "decoder_checkpoint_sha256": checkpoint_sha256,
                    "decoder_per_example_ref": reference,
                    "decoder_per_example_sha256": per_example_sha256,
                    "edit_id": None,
                    "edit_object": None,
                    "architecture": None,
                    "candidate_seed": None,
                    "edit_hash": None,
                },
                correct=correct,
                final_group_ids=final_group_ids,
            )
        )

    candidate_keys = [
        key for key in config.construct_edit_keys()
        if key.edit_kind == "dcand_crossfit"
    ]
    expected_edits = {
        _construct_edit_id(key.edit_kind, key.architecture, int(key.seed)): key
        for key in candidate_keys
    }
    candidate_rows = source.loc[
        (source["evaluation_family"] == "fresh_linear")
        & (source["edit_object"] == "dcand_crossfit")
    ].copy()
    if set(candidate_rows["edit_id"]) != set(expected_edits):
        raise RuntimeError("construct group statistics require the exact candidate-edit set")
    for edit_id in sorted(expected_edits):
        edit_key = expected_edits[edit_id]
        selected = candidate_rows.loc[candidate_rows["edit_id"] == edit_id]
        if len(selected) != 2 or set(selected["label"]) != {"target", "control"}:
            raise RuntimeError(f"construct candidate rows are incomplete for {edit_id}")
        edit_hashes = set(selected["edit_hash"])
        if len(edit_hashes) != 1:
            raise RuntimeError(f"construct candidate edit hash is inconsistent for {edit_id}")
        edit_hash = str(next(iter(edit_hashes)))
        if re.fullmatch(r"[0-9a-f]{64}", edit_hash) is None:
            raise RuntimeError(f"construct candidate edit hash is invalid for {edit_id}")
        for label in ("target", "control"):
            row = selected.loc[selected["label"] == label].iloc[0]
            if (
                row["architecture"] != edit_key.architecture
                or int(row["candidate_seed"]) != int(edit_key.seed)
            ):
                raise RuntimeError(f"construct candidate identity mismatch for {edit_id}")
            reference = _single_construct_artifact_value(
                row["per_example_refs"], field=f"{edit_id}/{label} per-example refs"
            )
            checkpoint_sha256 = _single_construct_artifact_value(
                row["decoder_checkpoint_hashes"],
                field=f"{edit_id}/{label} checkpoint hashes",
            )
            correct, observed_labels, per_example_sha256 = (
                _load_construct_decoder_per_example(
                    context,
                    reference=reference,
                    checkpoint_sha256=checkpoint_sha256,
                    final_example_ids=final_example_ids,
                    cell=cell,
                    legacy_layout=legacy_layout,
                    edit_id=edit_id,
                    label=label,
                )
            )
            _validate_construct_per_example_accuracy(
                correct,
                row["accuracy"],
                identity=f"{edit_id}/{label}",
            )
            if not torch.equal(observed_labels, baseline_labels[label]):
                raise RuntimeError(
                    f"construct decoder labels changed after edit for {edit_id}/{label}"
                )
            records.extend(
                _construct_group_records(
                    common={
                        **provenance,
                        "row_kind": "post_edit",
                        "label": label,
                        "decoder_checkpoint_sha256": checkpoint_sha256,
                        "decoder_per_example_ref": reference,
                        "decoder_per_example_sha256": per_example_sha256,
                        "edit_id": edit_id,
                        "edit_object": "dcand_crossfit",
                        "architecture": edit_key.architecture,
                        "candidate_seed": int(edit_key.seed),
                        "edit_hash": edit_hash,
                    },
                    correct=correct,
                    final_group_ids=final_group_ids,
                )
            )

    frame = pd.DataFrame(records)
    duplicates = frame.duplicated(list(_CONSTRUCT_GROUP_KEY_COLUMNS), keep=False)
    if duplicates.any():
        raise RuntimeError("construct group sufficient-statistic keys are not unique")
    return frame.sort_values(
        list(_CONSTRUCT_GROUP_KEY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)


def _materialize_construct_group_rows(
    rows: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    csv_path: Path,
    parquet_path: Path,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows).sort_values(
        list(_CONSTRUCT_GROUP_KEY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)
    existing_frames: list[pd.DataFrame] = []
    for path, loader in (
        (csv_path, pd.read_csv),
        (parquet_path, pd.read_parquet),
    ):
        if path.is_file():
            existing_frames.append(loader(path).reset_index(drop=True))
    if existing_frames:
        existing_statuses = set(existing_frames[0].get("status", pd.Series(dtype=str)))
        new_statuses = set(frame.get("status", pd.Series(dtype=str)))
        if existing_statuses == {"ok"} and new_statuses != {"ok"}:
            return existing_frames[0]
        if existing_statuses == {"ok"} and new_statuses == {"ok"}:
            try:
                for existing in existing_frames:
                    pd.testing.assert_frame_equal(
                        existing.astype(object).where(existing.notna(), None),
                        frame.astype(object).where(frame.notna(), None),
                        check_dtype=False,
                        check_like=False,
                    )
            except AssertionError as exc:
                raise RuntimeError(
                    "immutable construct group rows differ on resume"
                ) from exc
            if len(existing_frames) == 2:
                return pd.read_parquet(parquet_path).reset_index(drop=True)
    return materialize_rows(
        frame,
        csv_path=csv_path,
        parquet_path=parquet_path,
        key_columns=_CONSTRUCT_GROUP_KEY_COLUMNS,
    )


def materialize_construct_cell_status(
    context: RunContext,
    config: RevisionConfig,
    *,
    cell: ConstructCell,
    status: str,
    failure_stage: str,
    failure_reason: str,
) -> pd.DataFrame:
    """Persist one analysis-compatible explicit non-ok construct cell record."""

    if status not in {"failed", "nonestimable"}:
        raise ValueError("construct cell status must be failed or nonestimable")
    if not failure_stage.strip() or not failure_reason.strip():
        raise ValueError("construct cell failure stage and reason must be nonblank")
    layout = _construct_layout(context.run_dir, cell)
    if layout.group_rows_csv is None or layout.group_rows_parquet is None:
        raise RuntimeError("generic construct group-row paths are unavailable")
    row = {
        "run_id": context.run_id,
        "git_commit": _current_commit(),
        "config_hash": config.config_hash,
        **_construct_cell_record(cell),
        "row_kind": "cell_status",
        "evaluation_family": "fresh_linear",
        "decoder_seed": 0,
        "label": None,
        "group_id": None,
        "n_examples": 0,
        "correct_count": 0,
        "split_manifest_ref": None,
        "split_manifest_sha256": None,
        "decoder_checkpoint_sha256": None,
        "decoder_per_example_ref": None,
        "decoder_per_example_sha256": None,
        "edit_id": None,
        "edit_object": None,
        "architecture": None,
        "candidate_seed": None,
        "edit_hash": None,
        "status": status,
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
    }
    return _materialize_construct_group_rows(
        [row],
        csv_path=layout.group_rows_csv,
        parquet_path=layout.group_rows_parquet,
    )


def _write_immutable_json(path: Path, payload: Any) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if canonical_json(existing) != canonical_json(payload):
            raise RuntimeError(f"immutable JSON artifact differs on resume: {path.name}")
        return
    atomic_write_json(path, payload)


def _torch_payloads_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _torch_payloads_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _torch_payloads_equal(a, b) for a, b in zip(left, right)
        )
    return type(left) is type(right) and left == right


def _write_immutable_torch(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_file():
        existing = torch.load(path, map_location="cpu", weights_only=True)
        if not _torch_payloads_equal(existing, dict(payload)):
            raise RuntimeError(f"immutable tensor artifact differs on resume: {path.name}")
        return
    atomic_torch_save(path, dict(payload))


def _subset_manifest_rows(
    subdivision: GroupedSubdivision, subset: str
) -> list[dict[str, Any]]:
    return sorted(
        (row for row in subdivision.manifest if row["subset"] == subset),
        key=lambda row: row["position_in_subset"],
    )


def _metric_mean(records: list[dict[str, Any]], section: str, metric: str) -> float:
    values = [record[section][metric] for record in records]
    if any(value is None for value in values):
        raise ValueError(f"metric {section}.{metric} is undefined")
    return float(np.mean([float(value) for value in values]))


def _validate_rank_one_reconstruction(
    *,
    source_subsets: Mapping[str, torch.Tensor],
    saved_edited_subsets: Mapping[str, torch.Tensor],
    direction: torch.Tensor,
    validation_probe: torch.nn.Module,
    device: torch.device,
    tolerance: float,
) -> dict[str, Any]:
    if set(source_subsets) != set(saved_edited_subsets) or "final" not in source_subsets:
        raise ValueError("rank-one reconstruction subsets are incomplete")
    reconstructed: dict[str, torch.Tensor] = {}
    subset_differences: dict[str, float] = {}
    squared_error = 0.0
    element_count = 0
    for subset in source_subsets:
        reconstructed[subset], _ = rank_one_projection_edit(
            source_subsets[subset], direction
        )
        difference = reconstructed[subset] - saved_edited_subsets[subset].cpu().float()
        subset_differences[subset] = (
            float(difference.abs().max().item()) if difference.numel() else 0.0
        )
        squared_error += float(difference.double().square().sum().item())
        element_count += difference.numel()
    maximum_representation_difference = max(subset_differences.values(), default=0.0)
    saved_logits = evaluate_logits(
        validation_probe, saved_edited_subsets["final"], device
    ).detach().cpu().float()
    reconstructed_logits = evaluate_logits(
        validation_probe, reconstructed["final"], device
    ).detach().cpu().float()
    maximum_logit_difference = (
        float((saved_logits - reconstructed_logits).abs().max().item())
        if saved_logits.numel()
        else 0.0
    )
    if (
        maximum_representation_difference > tolerance
        or maximum_logit_difference > tolerance
    ):
        raise ValueError(
            "rank-one reconstruction differs from saved edit/logits: "
            f"representation={maximum_representation_difference}, "
            f"logits={maximum_logit_difference}, tolerance={tolerance}"
        )
    return {
        "tolerance": float(tolerance),
        "subset_maximum_absolute_differences": subset_differences,
        "maximum_absolute_representation_difference": maximum_representation_difference,
        "rms_representation_difference": math.sqrt(squared_error / max(1, element_count)),
        "maximum_absolute_logit_difference": maximum_logit_difference,
        "saved_final_logits_sha256": sha256_tensor(saved_logits),
        "reconstructed_final_logits_sha256": sha256_tensor(reconstructed_logits),
        "validation_probe_state_sha256": state_dict_sha256(_state_payload(validation_probe)),
    }


def _construct_candidate_device(
    architecture: str,
    accelerator_eligible_device: torch.device,
) -> torch.device:
    """Route only the permitted MLP candidate/Jacobian work to an accelerator."""

    return (
        accelerator_eligible_device
        if architecture == "mlp"
        else torch.device("cpu")
    )


def _run_construct_cell(
    context: RunContext,
    config: RevisionConfig,
    *,
    cell: ConstructCell,
    device: torch.device,
    legacy_layout: bool,
) -> dict[str, Any]:
    _validate_construct_cell(config, cell)
    layout = _construct_layout(context.run_dir, cell, legacy=legacy_layout)
    cell_record = _construct_cell_record(cell)
    scoped_cell_metadata = None if legacy_layout else cell_record
    _require_ok_report(context, "preflight_report.json")
    _require_ok_report(context, "matched_split_summary.json")
    reconstruction = _reconstruct_tasks(config)[cell.task]
    cache_pins = _preflight_cache_pins(context)
    caches = {
        tag: _load_cache(
            model_id=cell.model_id,
            task=cell.task,
            layer=cell.layer,
            tag=tag,
            reconstruction=reconstruction,
            expected_cache_sha256=cache_pins[
                (cell.model_id, cell.task, cell.layer, tag)
            ],
        )
        for tag in ("cand", "eval", "inter")
    }
    candidate_cache = caches["cand"]
    evaluator_cache = caches["eval"]
    intervention_cache = caches["inter"]
    mean = candidate_cache.X.float().mean(dim=0, keepdim=True)
    standard_deviation = candidate_cache.X.float().std(dim=0, keepdim=True).clamp_min(1.0e-6)
    standardization_path = layout.artifact_root / "standardization.pt"
    if standardization_path.is_file():
        saved_standardization = torch.load(
            standardization_path, map_location="cpu", weights_only=True
        )
        if not torch.equal(saved_standardization["mean"], mean) or not torch.equal(
            saved_standardization["standard_deviation"], standard_deviation
        ):
            raise RuntimeError("construct standardization differs on resume")
    else:
        atomic_torch_save(
            standardization_path,
            _construct_payload({
                "mean": mean,
                "standard_deviation": standard_deviation,
                "candidate_cache_sha256": candidate_cache.selection.cache_sha256,
            }, cell=cell, legacy=legacy_layout),
        )

    def standardize(X: torch.Tensor) -> torch.Tensor:
        return ((X.float() - mean) / standard_deviation).contiguous()

    X_candidate = standardize(candidate_cache.X)
    X_evaluator = standardize(evaluator_cache.X)
    X_intervention = standardize(intervention_cache.X)
    construct_seed = int(config.raw["reproducibility"]["master_seed"])
    subdivision = subdivide_phase2_intervention(
        reconstruction.folds["intervention"],
        seed=construct_seed,
        task_name=cell.task,
    )
    indices = _subdivision_indices(reconstruction.folds["intervention"], subdivision)
    named_groups = {
        "candidate_attacker": set(_group_ids(reconstruction, "candidate")),
        "evaluator": set(_group_ids(reconstruction, "evaluator")),
        "direction_fit": {
            row["group_id"] for row in _subset_manifest_rows(subdivision, "direction_fit")
        },
        "fresh_decoder_fit": {
            row["group_id"] for row in _subset_manifest_rows(subdivision, "fresh_decoder_fit")
        },
        "orientation_calibration": {
            row["group_id"] for row in _subset_manifest_rows(subdivision, "orientation_calibration")
        },
        "final_test": {
            row["group_id"] for row in _subset_manifest_rows(subdivision, "final_test")
        },
        "unused_phase2_test": set(_group_ids(reconstruction, "test")),
    }
    assert_disjoint_named_groups(named_groups)
    split_payload = _construct_payload({
        "source_data_hash": reconstruction.all_data_hash,
        "phase2_fold_hashes": reconstruction.fold_hashes,
        "phase2_manifest": reconstruction.manifest,
        "intervention_subdivision": subdivision.manifest,
        "intervention_subdivision_hashes": subdivision.subset_hashes,
        "intervention_subdivision_diagnostics": subdivision.diagnostics,
        "split_seed": construct_seed,
        "group_disjointness": True,
    }, cell=cell, legacy=legacy_layout)
    split_manifest_path = layout.artifact_root / "split_manifest.json"
    _write_immutable_json(split_manifest_path, split_payload)
    split_manifest_ref = _relative(split_manifest_path, context.run_dir)

    X_direction = X_intervention[indices["direction_fit"]]
    X_decoder = X_intervention[indices["fresh_decoder_fit"]]
    X_calibration = X_intervention[indices["orientation_calibration"]]
    X_final = X_intervention[indices["final_test"]]
    final_example_ids = [
        row["example_id"] for row in _subset_manifest_rows(subdivision, "final_test")
    ]
    labels = {
        "target": {
            "candidate": candidate_cache.zc,
            "evaluator": evaluator_cache.zc,
            "decoder": intervention_cache.zc[indices["fresh_decoder_fit"]],
            "calibration": intervention_cache.zc[indices["orientation_calibration"]],
            "final": intervention_cache.zc[indices["final_test"]],
        },
        "control": {
            "candidate": candidate_cache.ze,
            "evaluator": evaluator_cache.ze,
            "decoder": intervention_cache.ze[indices["fresh_decoder_fit"]],
            "calibration": intervention_cache.ze[indices["orientation_calibration"]],
            "final": intervention_cache.ze[indices["final_test"]],
        },
    }
    decoder_group_strings = [
        row["group_id"] for row in _subset_manifest_rows(subdivision, "fresh_decoder_fit")
    ]
    group_numbers = {
        value: index for index, value in enumerate(sorted(set(decoder_group_strings)))
    }
    decoder_groups = torch.tensor(
        [group_numbers[value] for value in decoder_group_strings], dtype=torch.long
    )

    hyperparameters: dict[str, Any] = {}
    selections = {}
    for label_name, selection_seed in (("target", construct_seed), ("control", construct_seed + 1)):
        selection = select_linear_hyperparameters(
            X=X_decoder,
            y=labels[label_name]["decoder"],
            groups=decoder_groups,
            learning_rates=[0.0003, 0.001, 0.003],
            weight_decays=[0.0, 0.001, 0.01, 0.1],
            epochs=50,
            batch_size=256,
            patience=5,
            seed=selection_seed,
            device=torch.device("cpu"),
        )
        selections[label_name] = selection
        hyperparameters[label_name] = {
            "linear_selected": asdict(selection.selected_spec),
            "linear_records": selection.records,
            "inner_train_indices": selection.train_indices.tolist(),
            "inner_validation_indices": selection.validation_indices.tolist(),
            "inner_train_group_hash": sha256_json(
                sorted(set(decoder_groups[selection.train_indices].tolist()))
            ),
            "inner_validation_group_hash": sha256_json(
                sorted(set(decoder_groups[selection.validation_indices].tolist()))
            ),
            "mlp_spec": asdict(
                DecoderSpec(
                    architecture="mlp",
                    hidden_dim=256,
                    learning_rate=0.001,
                    weight_decay=0.01,
                    epochs=50,
                    batch_size=256,
                    patience=5,
                )
            ),
            "mlp_seeds": [0, 1, 2],
        }
    hyperparameter_path = layout.artifact_root / "hyperparameter_selection.json"
    _write_immutable_json(
        hyperparameter_path,
        _construct_payload(hyperparameters, cell=cell, legacy=legacy_layout),
    )

    mlp_spec = DecoderSpec(
        architecture="mlp",
        hidden_dim=256,
        learning_rate=0.001,
        weight_decay=0.01,
        epochs=50,
        batch_size=256,
        patience=5,
    )
    baseline_metrics: dict[str, dict[str, Any]] = {}
    for family, seeds in (("fresh_linear", [0]), ("fresh_mlp", [0, 1, 2])):
        baseline_metrics[family] = {}
        for label_name in ("target", "control"):
            selection = selections[label_name]
            spec = selection.selected_spec if family == "fresh_linear" else mlp_spec
            records = []
            hashes = []
            per_example_refs = []
            for seed in seeds:
                metric, checkpoint_hash, history, best_epoch, per_example_path = _fresh_checkpoint(
                    layout.checkpoint_root / "fresh"
                    / "unedited" / f"{family}-{label_name}-seed{seed}.pt",
                    X_train=X_decoder[selection.train_indices],
                    y_train=labels[label_name]["decoder"][selection.train_indices],
                    X_validation=X_decoder[selection.validation_indices],
                    y_validation=labels[label_name]["decoder"][selection.validation_indices],
                    X_final=X_final,
                    y_final=labels[label_name]["final"],
                    final_example_ids=final_example_ids,
                    spec=spec,
                    seed=seed,
                    device=torch.device("cpu"),
                    metadata=_construct_payload({
                        "edit_id": "unedited",
                        "family": family,
                        "label": label_name,
                        "split_hashes": subdivision.subset_hashes,
                    }, cell=cell, legacy=legacy_layout),
                )
                records.append({**metric, "best_epoch": best_epoch, "history": history})
                hashes.append(checkpoint_hash)
                per_example_refs.append(_relative(per_example_path, context.run_dir))
            baseline_metrics[family][label_name] = {
                "metrics": {
                    metric: float(np.mean([record[metric] for record in records]))
                    for metric in ("accuracy", "balanced_accuracy", "auc", "log_loss")
                },
                "checkpoint_hashes": hashes,
                "per_example_refs": per_example_refs,
                "seed_records": records,
            }
    baseline_path = layout.artifact_root / "fresh_decoder_baselines.json"
    _write_immutable_json(
        baseline_path,
        _construct_payload(baseline_metrics, cell=cell, legacy=legacy_layout),
    )
    construct_artifact_provenance = {
        "split_manifest_ref": split_manifest_ref,
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "split_hashes": dict(subdivision.subset_hashes),
        "fresh_decoder_baseline_ref": _relative(baseline_path, context.run_dir),
        "fresh_decoder_baseline_sha256": sha256_file(baseline_path),
        "hyperparameter_selection_ref": _relative(
            hyperparameter_path, context.run_dir
        ),
        "hyperparameter_selection_sha256": sha256_file(hyperparameter_path),
    }

    attackers = _construct_attackers(
        context,
        X_candidate=X_candidate,
        zc_candidate=labels["target"]["candidate"],
        ze_candidate=labels["control"]["candidate"],
        cache_hash=candidate_cache.selection.cache_sha256,
        pair_seeds=config.pair_seeds,
        device=torch.device("cpu"),
        checkpoint_root=layout.checkpoint_root,
        cell_metadata=scoped_cell_metadata,
    )
    evaluators = _construct_evaluators(
        context,
        X_evaluator=X_evaluator,
        zc_evaluator=labels["target"]["evaluator"],
        ze_evaluator=labels["control"]["evaluator"],
        cache_hash=evaluator_cache.selection.cache_sha256,
        device=torch.device("cpu"),
        checkpoint_root=layout.checkpoint_root,
        cell_metadata=scoped_cell_metadata,
    )
    evaluator_hashes = {
        "target": [state_dict_sha256(_state_payload(evaluator.probes.zc_probe)) for evaluator in evaluators],
        "control": [state_dict_sha256(_state_payload(evaluator.probes.ze_probe)) for evaluator in evaluators],
    }

    edit_keys = config.construct_edit_keys()
    shard_keys = [layout.shard_key(key) for key in edit_keys]
    completed_edits = context.completed_keys(
        layout.shard_experiment,
        expected_keys=shard_keys,
    )
    fatal_errors: list[str] = []
    for edit_key in edit_keys:
        edit_tuple = tuple(edit_key)
        shard_key = layout.shard_key(edit_tuple)
        if shard_key in completed_edits:
            continue
        edit_id = _construct_edit_id(*edit_key)
        LOGGER.info("construct check: %s", edit_id)
        candidate_checkpoint_hash: str | None = None
        direction_diagnostics: dict[str, Any] = {}
        direction_path: Path | None = None
        try:
            if edit_key.edit_kind == "alterrep":
                attacker = attackers[int(edit_key.seed)]
                candidate_checkpoint_hash = attacker.target_checkpoint_hash
                edited = {
                    "decoder": alterrep_edit(
                        X_decoder,
                        labels["target"]["decoder"],
                        attacker.target_probe,
                        device=torch.device("cpu"),
                        alpha=1.0,
                    ),
                    "calibration": alterrep_edit(
                        X_calibration,
                        labels["target"]["calibration"],
                        attacker.target_probe,
                        device=torch.device("cpu"),
                        alpha=1.0,
                    ),
                    "final": alterrep_edit(
                        X_final,
                        labels["target"]["final"],
                        attacker.target_probe,
                        device=torch.device("cpu"),
                        alpha=1.0,
                    ),
                }
                recipe_path = layout.artifact_root / "edits" / f"{edit_id}.pt"
                if recipe_path.is_file():
                    saved = torch.load(recipe_path, map_location="cpu", weights_only=True)
                    for subset, value in edited.items():
                        if not torch.equal(saved[subset], value):
                            raise RuntimeError("AlterRep edited shard differs on resume")
                else:
                    atomic_torch_save(
                        recipe_path,
                        _construct_payload({
                            **edited,
                            "source_cache_sha256": intervention_cache.selection.cache_sha256,
                            "dtype": str(X_decoder.dtype),
                            "attacker_checkpoint_hash": candidate_checkpoint_hash,
                        }, cell=cell, legacy=legacy_layout),
                    )
            else:
                candidate_device = _construct_candidate_device(
                    edit_key.architecture,
                    device,
                )
                candidate, candidate_checkpoint_hash = _construct_candidate(
                    context,
                    architecture=edit_key.architecture,
                    seed=int(edit_key.seed),
                    X_candidate=X_candidate,
                    labels=labels["target"]["candidate"],
                    cache_hash=candidate_cache.selection.cache_sha256,
                    device=candidate_device,
                    checkpoint_root=layout.checkpoint_root,
                    cell_metadata=scoped_cell_metadata,
                )
                direction, direction_diagnostics = candidate_rank_one_direction(
                    candidate, X_direction, device=candidate_device
                )
                direction_diagnostics["candidate_compute_device"] = str(
                    candidate_device
                )
                edited = {}
                residuals = {}
                for subset, X_subset in (
                    ("decoder", X_decoder),
                    ("calibration", X_calibration),
                    ("final", X_final),
                ):
                    edited[subset], residuals[subset] = rank_one_projection_edit(
                        X_subset, direction
                    )
                direction_diagnostics["projection_residuals"] = residuals
                source_subsets = {
                    "decoder": X_decoder,
                    "calibration": X_calibration,
                    "final": X_final,
                }
                reconstruction_validation = _validate_rank_one_reconstruction(
                    source_subsets=source_subsets,
                    saved_edited_subsets=edited,
                    direction=direction,
                    validation_probe=evaluators[0].probes.zc_probe,
                    device=torch.device("cpu"),
                    tolerance=1.0e-6,
                )
                direction_diagnostics["reconstruction_validation"] = (
                    reconstruction_validation
                )
                direction_path = layout.artifact_root / "directions" / f"{edit_id}.pt"
                edited_subset_names = {
                    "decoder": "fresh_decoder_fit",
                    "calibration": "orientation_calibration",
                    "final": "final_test",
                }
                _write_immutable_torch(
                    direction_path,
                    _construct_payload({
                        "schema_version": 2,
                        "direction": direction.detach().cpu().float(),
                        "diagnostics": direction_diagnostics,
                        "candidate_checkpoint_hash": candidate_checkpoint_hash,
                        "source_cache_sha256": intervention_cache.selection.cache_sha256,
                        "source_cache_data_hash": intervention_cache.selection.data_hash,
                        "source_standardization_ref": _relative(
                            standardization_path, context.run_dir
                        ),
                        "source_standardization_sha256": sha256_file(
                            standardization_path
                        ),
                        "direction_fit_example_ids": [
                            row["example_id"]
                            for row in _subset_manifest_rows(subdivision, "direction_fit")
                        ],
                        "direction_fit_source_indices": indices["direction_fit"].tolist(),
                        "edited_subset_example_ids": {
                            subset: [
                                row["example_id"]
                                for row in _subset_manifest_rows(subdivision, source_name)
                            ]
                            for subset, source_name in edited_subset_names.items()
                        },
                        "edited_subset_source_indices": {
                            subset: indices[source_name].tolist()
                            for subset, source_name in edited_subset_names.items()
                        },
                        "projection_code": (
                            "float32(x - (x @ d) * d); d = d / ||d||_2; "
                            "projection and residual accumulated in float64"
                        ),
                        "projection_code_version": "reviewer_revision.rank_one_projection.v2",
                        "source_representation_dtype": str(X_direction.dtype),
                        "direction_dtype": str(direction.dtype),
                        "reconstruction_validation": reconstruction_validation,
                    }, cell=cell, legacy=legacy_layout),
                )

            edit_hash = sha256_json(
                {subset: sha256_tensor(value) for subset, value in edited.items()}
            )
            fixed_records = {"target": [], "control": []}
            for evaluator_index, evaluator in enumerate(evaluators):
                for label_name, probe in (
                    ("target", evaluator.probes.zc_probe),
                    ("control", evaluator.probes.ze_probe),
                ):
                    result = evaluate_fixed_evaluator_edit(
                        probe=probe,
                        X_calibration_pre=X_calibration,
                        X_calibration_post=edited["calibration"],
                        calibration_labels=labels[label_name]["calibration"],
                        X_final_pre=X_final,
                        X_final_post=edited["final"],
                        final_labels=labels[label_name]["final"],
                        device=torch.device("cpu"),
                    )
                    if (
                        direction_path is not None
                        and evaluator_index == 0
                        and label_name == "target"
                    ):
                        persisted_logits = torch.tensor(
                            result["per_example"]["final_post"]["logits"],
                            dtype=torch.float32,
                        )
                        persisted_hash = sha256_tensor(persisted_logits)
                        expected_hash = direction_diagnostics[
                            "reconstruction_validation"
                        ]["saved_final_logits_sha256"]
                        if persisted_hash != expected_hash:
                            raise RuntimeError(
                                "rank-one reconstructed logits do not match saved "
                                "fixed-evaluator logits"
                            )
                        result["rank_one_reconstruction_validation"] = {
                            "passed": True,
                            "direction_artifact_ref": _relative(
                                direction_path, context.run_dir
                            ),
                            "persisted_final_logits_sha256": persisted_hash,
                            "maximum_absolute_logit_difference": direction_diagnostics[
                                "reconstruction_validation"
                            ]["maximum_absolute_logit_difference"],
                            "tolerance": 1.0e-6,
                        }
                    result["calibration_example_ids"] = [
                        row["example_id"]
                        for row in _subset_manifest_rows(subdivision, "orientation_calibration")
                    ]
                    result["final_example_ids"] = [
                        row["example_id"]
                        for row in _subset_manifest_rows(subdivision, "final_test")
                    ]
                    if not legacy_layout:
                        result["cell"] = cell_record
                    output_path = (
                        layout.array_root / edit_id
                        / f"evaluator{evaluator_index}-{label_name}.json"
                    )
                    _write_immutable_json(output_path, result)
                    result["per_example_ref"] = _relative(output_path, context.run_dir)
                    fixed_records[label_name].append(result)

            rows: list[dict[str, Any]] = []
            for label_name in ("target", "control"):
                records = fixed_records[label_name]
                accuracy = _metric_mean(records, "final_post_raw", "accuracy")
                pre_accuracy = _metric_mean(records, "final_pre_raw", "accuracy")
                oriented_accuracy = _metric_mean(records, "final_post_oriented", "accuracy")
                recovery_ratio = _ratio(accuracy, pre_accuracy)
                rows.append(
                    {
                        **_construct_row_base(
                            context=context,
                            config=config,
                            cell=cell,
                            edit_id=edit_id,
                            edit_kind=edit_key.edit_kind,
                            architecture=edit_key.architecture,
                            seed=int(edit_key.seed),
                            evaluation_family="fixed",
                            label=label_name,
                            edit_hash=edit_hash,
                            candidate_checkpoint_hash=candidate_checkpoint_hash,
                            evaluator_checkpoint_hashes=evaluator_hashes[label_name],
                            split_manifest_ref=split_manifest_ref,
                            artifact_provenance=construct_artifact_provenance,
                        ),
                        "accuracy": accuracy,
                        "pre_edit_accuracy": pre_accuracy,
                        "balanced_accuracy": _metric_mean(records, "final_post_raw", "balanced_accuracy"),
                        "calibrated_orientation_accuracy": oriented_accuracy,
                        "max_accuracy_sensitivity": _metric_mean(
                            records, "final_post_raw", "complement_sensitivity_accuracy"
                        ),
                        "auc": _metric_mean(records, "final_post_raw", "auc"),
                        "log_loss": _metric_mean(records, "final_post_raw", "log_loss"),
                        "orientation_adjusted_auc": _metric_mean(records, "final_post_oriented", "auc"),
                        "orientation_adjusted_log_loss": _metric_mean(
                            records, "final_post_oriented", "log_loss"
                        ),
                        "C_raw": float(np.mean([record["C_raw"] for record in records])),
                        "C_orientation_calibrated": float(
                            np.mean([record["C_orientation"] for record in records])
                        ),
                        "target_recovery_ratio": (
                            recovery_ratio if label_name == "target" else None
                        ),
                        "control_retention_ratio": (
                            recovery_ratio if label_name == "control" else None
                        ),
                        "orientation_signs": [record["orientation"]["sign"] for record in records],
                        "per_example_refs": [record["per_example_ref"] for record in records],
                        "direction_diagnostics": direction_diagnostics,
                    }
                )

            for family, seeds in (("fresh_linear", [0]), ("fresh_mlp", [0, 1, 2])):
                for label_name in ("target", "control"):
                    selection = selections[label_name]
                    spec = selection.selected_spec if family == "fresh_linear" else mlp_spec
                    metric_records = []
                    checkpoint_hashes = []
                    curve_refs = []
                    per_example_refs = []
                    for seed in seeds:
                        checkpoint_path = (
                            layout.checkpoint_root / "fresh"
                            / edit_id / f"{family}-{label_name}-seed{seed}.pt"
                        )
                        metric, checkpoint_hash, history, best_epoch, per_example_path = _fresh_checkpoint(
                            checkpoint_path,
                            X_train=edited["decoder"][selection.train_indices],
                            y_train=labels[label_name]["decoder"][selection.train_indices],
                            X_validation=edited["decoder"][selection.validation_indices],
                            y_validation=labels[label_name]["decoder"][selection.validation_indices],
                            X_final=edited["final"],
                            y_final=labels[label_name]["final"],
                            final_example_ids=final_example_ids,
                            spec=spec,
                            seed=seed,
                            device=torch.device("cpu"),
                            metadata=_construct_payload({
                                "edit_id": edit_id,
                                "edit_hash": edit_hash,
                                "family": family,
                                "label": label_name,
                                "split_hashes": subdivision.subset_hashes,
                                "hyperparameter_selection_sha256": sha256_file(hyperparameter_path),
                            }, cell=cell, legacy=legacy_layout),
                        )
                        metric_records.append({**metric, "best_epoch": best_epoch})
                        checkpoint_hashes.append(checkpoint_hash)
                        per_example_refs.append(_relative(per_example_path, context.run_dir))
                        curve_path = checkpoint_path.with_suffix(".curve.json")
                        _write_immutable_json(
                            curve_path,
                            _construct_payload({
                                "history": history,
                                "best_epoch": best_epoch,
                                "checkpoint_sha256": checkpoint_hash,
                            }, cell=cell, legacy=legacy_layout),
                        )
                        curve_refs.append(_relative(curve_path, context.run_dir))
                    metrics = {
                        metric: float(np.mean([record[metric] for record in metric_records]))
                        for metric in ("accuracy", "balanced_accuracy", "auc", "log_loss")
                    }
                    baseline = baseline_metrics[family][label_name]["metrics"]
                    recovery_ratio = _ratio(metrics["accuracy"], baseline["accuracy"])
                    damage = compute_damage_score(
                        baseline["accuracy"], metrics["accuracy"], 1.0, 1.0
                    )
                    rows.append(
                        {
                            **_construct_row_base(
                                context=context,
                                config=config,
                                cell=cell,
                                edit_id=edit_id,
                                edit_kind=edit_key.edit_kind,
                                architecture=edit_key.architecture,
                                seed=int(edit_key.seed),
                                evaluation_family=family,
                                label=label_name,
                                edit_hash=edit_hash,
                                candidate_checkpoint_hash=candidate_checkpoint_hash,
                                evaluator_checkpoint_hashes=evaluator_hashes[label_name],
                                split_manifest_ref=split_manifest_ref,
                                artifact_provenance=construct_artifact_provenance,
                            ),
                            "accuracy": metrics["accuracy"],
                            "pre_edit_accuracy": baseline["accuracy"],
                            "balanced_accuracy": metrics["balanced_accuracy"],
                            "calibrated_orientation_accuracy": metrics["accuracy"],
                            "max_accuracy_sensitivity": max(metrics["accuracy"], 1.0 - metrics["accuracy"]),
                            "auc": metrics["auc"],
                            "log_loss": metrics["log_loss"],
                            "orientation_adjusted_auc": metrics["auc"],
                            "orientation_adjusted_log_loss": metrics["log_loss"],
                            "C_raw": damage.C if damage.C is not None else 0.0,
                            "C_orientation_calibrated": damage.C if damage.C is not None else 0.0,
                            "target_recovery_ratio": (
                                recovery_ratio if label_name == "target" else None
                            ),
                            "control_retention_ratio": (
                                recovery_ratio if label_name == "control" else None
                            ),
                            "decoder_checkpoint_hashes": checkpoint_hashes,
                            "training_curve_refs": curve_refs,
                            "per_example_refs": per_example_refs,
                            "best_epochs": [record["best_epoch"] for record in metric_records],
                            "selected_hyperparameters": asdict(spec),
                            "direction_diagnostics": direction_diagnostics,
                        }
                    )
            context.write_json_shard(
                layout.shard_experiment,
                shard_key,
                _construct_payload({
                    "status": "ok",
                    "edit_id": edit_id,
                    "edit_hash": edit_hash,
                    "rows": rows,
                    "direction_diagnostics": direction_diagnostics,
                }, cell=cell, legacy=legacy_layout),
            )
            completed_edits.add(shard_key)
        except Exception as error:  # noqa: BLE001 - persist explicit edit failure
            fatal_errors.append(f"{edit_id}: {type(error).__name__}: {error}")
            failure_rows = []
            for family in ("fixed", "fresh_linear", "fresh_mlp"):
                for label_name in ("target", "control"):
                    failure_rows.append(
                        {
                            **_construct_row_base(
                                context=context,
                                config=config,
                                cell=cell,
                                edit_id=edit_id,
                                edit_kind=edit_key.edit_kind,
                                architecture=edit_key.architecture,
                                seed=int(edit_key.seed),
                                evaluation_family=family,
                                label=label_name,
                                edit_hash="failed",
                                candidate_checkpoint_hash=candidate_checkpoint_hash,
                                evaluator_checkpoint_hashes=evaluator_hashes[label_name],
                                split_manifest_ref=split_manifest_ref,
                                artifact_provenance=construct_artifact_provenance,
                            ),
                            "status": "failed",
                            "failure_stage": "construct_edit_or_evaluation",
                            "failure_reason": f"{type(error).__name__}: {error}",
                            "pre_edit_accuracy": baseline_metrics.get(family, {})
                            .get(label_name, {})
                            .get("metrics", {})
                            .get("accuracy"),
                        }
                    )
            context.write_json_shard(
                layout.shard_experiment,
                shard_key,
                _construct_payload({
                    "status": "failed",
                    "edit_id": edit_id,
                    "edit_hash": None,
                    "rows": failure_rows,
                    "failure_stage": "construct_edit_or_evaluation",
                    "failure_reason": f"{type(error).__name__}: {error}",
                }, cell=cell, legacy=legacy_layout),
            )
            # Materialize the explicit failure rows for this invocation.
            # Resume excludes this top-level failed payload and retries it.
            completed_edits.add(shard_key)

    if completed_edits != set(shard_keys):
        raise RuntimeError("construct check edit shards are incomplete")
    construct_rows = []
    for edit_key in edit_keys:
        construct_rows.extend(
            context.load_json_shard(
                layout.shard_experiment,
                layout.shard_key(edit_key),
            )["rows"]
        )
    expected_row_keys = _construct_expected_keys(config)
    frame = materialize_rows(
        construct_rows,
        csv_path=layout.rows_csv,
        parquet_path=layout.rows_parquet,
        key_columns=("edit_id", "evaluation_family", "label"),
    )
    observed_keys = {
        tuple(row)
        for row in frame[["edit_id", "evaluation_family", "label"]].itertuples(index=False, name=None)
    }
    if observed_keys != set(expected_row_keys):
        raise RuntimeError("construct check exact row-key coverage failed")
    if fatal_errors or (frame["status"] == "failed").any():
        context.update_manifest(
            {layout.manifest_key: {"status": "failed", "errors": fatal_errors[:20]}}
        )
        raise RuntimeError(f"construct check contains {len(fatal_errors)} failed edits")
    construct_provenance = _regenerate_construct_provenance(context, frame)
    group_frame: pd.DataFrame | None = None
    if not legacy_layout:
        if layout.group_rows_csv is None or layout.group_rows_parquet is None:
            raise RuntimeError("generic construct group-row paths are unavailable")
        group_frame = build_construct_group_rows(
            context,
            config,
            cell=cell,
            legacy_layout=False,
        )
        group_frame = _materialize_construct_group_rows(
            group_frame,
            csv_path=layout.group_rows_csv,
            parquet_path=layout.group_rows_parquet,
        )
    expected_edit_ids = [_construct_edit_id(*key) for key in edit_keys]
    summary = summarize_construct_check(
        frame,
        expected_keys=expected_row_keys,
        key_columns=("edit_id", "evaluation_family", "label"),
        expected_edit_ids=expected_edit_ids,
        raw_row_file_sha256=sha256_file(layout.rows_parquet),
        generating_git_commit=_current_commit(),
    )
    summary["status"] = "ok"
    if group_frame is not None:
        summary["construct_group_rows_ref"] = _relative(
            layout.group_rows_parquet, context.run_dir
        )
        summary["construct_group_rows_sha256"] = sha256_file(
            layout.group_rows_parquet
        )
        summary["construct_group_row_count"] = len(group_frame)
    summary.update(
        {
            "cell": layout.summary_cell,
            "confidence_interval_method": "descriptive full distributions across prespecified edits",
            "cluster_unit": "candidate_edit_with_five_evaluator_bags",
            "bootstrap_seed": int(config.raw["reproducibility"]["bootstrap_seed"]),
            "bootstrap_draws": int(config.raw["reproducibility"]["bootstrap_draws"]),
            "permutation_seed": int(config.raw["reproducibility"]["permutation_seed"]),
            **construct_provenance,
        }
    )
    atomic_write_json(layout.summary_path, summary)
    context.update_manifest(
        {
            layout.manifest_key: {
                "status": "ok",
                "edits": len(edit_keys),
                "rows": len(frame),
                **(
                    {
                        "construct_group_rows": len(group_frame),
                        "construct_group_rows_sha256": sha256_file(
                            layout.group_rows_parquet
                        ),
                    }
                    if group_frame is not None
                    else {}
                ),
            }
        }
    )
    return summary


def run_construct_cell(
    context: RunContext,
    config: RevisionConfig,
    *,
    cell: ConstructCell,
    device: torch.device,
) -> dict[str, Any]:
    """Run one namespaced middle-layer construct cell."""

    try:
        return _run_construct_cell(
            context,
            config,
            cell=cell,
            device=device,
            legacy_layout=False,
        )
    except Exception as error:
        materialize_construct_cell_status(
            context,
            config,
            cell=cell,
            status="failed",
            failure_stage="construct_cell_execution",
            failure_reason=f"{type(error).__name__}: {error}",
        )
        raise


def run_construct_check(
    context: RunContext,
    config: RevisionConfig,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Run the original Qwen/SST-2 pilot with its legacy artifact contract."""

    return _run_construct_cell(
        context,
        config,
        cell=_PILOT_CONSTRUCT_CELL,
        device=device,
        legacy_layout=True,
    )


def run_analysis(context: RunContext, config: RevisionConfig) -> dict[str, Any]:
    stored_matched = json.loads(
        (context.run_dir / "matched_split_summary.json").read_text(encoding="utf-8")
    )
    stored_epsilon = json.loads(
        (context.run_dir / "epsilon_sweep_summary.json").read_text(encoding="utf-8")
    )
    stored_construct = json.loads(
        (context.run_dir / "construct_check_summary.json").read_text(encoding="utf-8")
    )
    if any(
        summary.get("status") != "ok"
        for summary in (stored_matched, stored_epsilon, stored_construct)
    ):
        raise RuntimeError("analysis requires three successful experiment summaries")
    if any(
        summary.get("failed_units", 0)
        for summary in (stored_matched, stored_epsilon, stored_construct)
    ):
        raise RuntimeError("analysis refused summaries with failed units")

    benchmark = _load_benchmark_report(context)
    grid = benchmark["selected_grid"]
    matched_rows = pd.read_parquet(context.run_dir / "matched_split_rows.parquet")
    epsilon_rows = pd.read_parquet(context.run_dir / "epsilon_sweep_rows.parquet")
    construct_rows = pd.read_parquet(context.run_dir / "construct_check_rows.parquet")

    regenerated_matched = summarize_matched_split(
        matched_rows,
        expected_keys=config.matched_split_row_keys(grid),
        key_columns=("model_key", "task", "layer", "pair_seed", "method", "condition"),
        expected_blocks=12,
        bootstrap_draws=int(config.raw["reproducibility"]["bootstrap_draws"]),
        bootstrap_seed=int(config.raw["reproducibility"]["bootstrap_seed"]),
        permutation_seed=int(config.raw["reproducibility"]["permutation_seed"]),
        raw_row_file_sha256=sha256_file(context.run_dir / "matched_split_rows.parquet"),
        generating_git_commit=_current_commit(),
    )
    regenerated_matched.update({"status": "ok", "selected_grid": grid})

    epsilon_expected_keys = _epsilon_expected_keys(
        config, scope=benchmark["epsilon_scope"], grid=grid
    )
    regenerated_epsilon = summarize_epsilon_sweep(
        epsilon_rows,
        expected_keys=epsilon_expected_keys,
        key_columns=(
            "model_key",
            "task",
            "layer",
            "pair_seed",
            "method",
            "epsilon",
            "condition",
        ),
        raw_row_file_sha256=sha256_file(context.run_dir / "epsilon_sweep_rows.parquet"),
        generating_git_commit=_current_commit(),
    )
    epsilon_comparison = _validate_epsilon_baseline_against_matched(
        matched_rows, epsilon_rows, expected_rows=240
    )
    required_epsilon_rows = epsilon_rows.loc[
        epsilon_rows["epsilon_scope"] == "required_middle"
    ]
    epsilon_half_pattern = _summarize_epsilon_half_pattern(required_epsilon_rows)
    regenerated_epsilon.update(
        {
            "status": "ok",
            "scope": benchmark["epsilon_scope"],
            "required_middle_rows": len(required_epsilon_rows),
            "baseline_ceiling_reproduced": True,
            "epsilon_half_ceiling_pattern": epsilon_half_pattern,
            "epsilon_half_experiment_a_reproduction": epsilon_comparison,
        }
    )

    construct_expected_keys = _construct_expected_keys(config)
    construct_edit_ids = [_construct_edit_id(*key) for key in config.construct_edit_keys()]
    regenerated_construct = summarize_construct_check(
        construct_rows,
        expected_keys=construct_expected_keys,
        key_columns=("edit_id", "evaluation_family", "label"),
        expected_edit_ids=construct_edit_ids,
        raw_row_file_sha256=sha256_file(context.run_dir / "construct_check_rows.parquet"),
        generating_git_commit=_current_commit(),
    )
    regenerated_construct_provenance = _regenerate_construct_provenance(
        context, construct_rows
    )
    regenerated_construct.update(
        {
            "status": "ok",
            "cell": {"model_key": "qwen", "task": "sst2", "layer": 14},
            "confidence_interval_method": "descriptive full distributions across prespecified edits",
            "cluster_unit": "candidate_edit_with_five_evaluator_bags",
            "bootstrap_seed": int(config.raw["reproducibility"]["bootstrap_seed"]),
            "bootstrap_draws": int(config.raw["reproducibility"]["bootstrap_draws"]),
            "permutation_seed": int(config.raw["reproducibility"]["permutation_seed"]),
            **regenerated_construct_provenance,
        }
    )

    for name, stored, regenerated in (
        ("matched_split", stored_matched, regenerated_matched),
        ("epsilon_sweep", stored_epsilon, regenerated_epsilon),
        ("construct_check", stored_construct, regenerated_construct),
    ):
        if canonical_json(stored) != canonical_json(regenerated):
            differing = sorted(
                key
                for key in set(stored) | set(regenerated)
                if canonical_json(stored.get(key)) != canonical_json(regenerated.get(key))
            )
            raise RuntimeError(
                f"{name} summary does not regenerate from saved rows; "
                f"differing fields={differing}"
            )

    matched = regenerated_matched
    epsilon = regenerated_epsilon
    construct = regenerated_construct
    if matched["included_units"]["model_task_blocks"] != 12:
        raise RuntimeError("matched/split analysis does not contain 12 model-task blocks")
    if epsilon["required_middle_rows"] != 2400:
        raise RuntimeError("epsilon analysis does not contain all 2,400 required rows")
    if construct["included_units"]["edits"] != 65:
        raise RuntimeError("construct analysis does not contain all 65 edits")
    if not epsilon["epsilon_zero_integrity"]["passed"]:
        raise RuntimeError("epsilon-zero integrity is false")

    candidate_rows = construct_rows.loc[
        construct_rows["edit_object"].astype(str) == "dcand_crossfit"
    ]
    fixed_target = candidate_rows.loc[
        (candidate_rows["evaluation_family"] == "fixed")
        & (candidate_rows["label"] == "target")
    ]
    fresh_linear_target = candidate_rows.loc[
        (candidate_rows["evaluation_family"] == "fresh_linear")
        & (candidate_rows["label"] == "target")
    ]
    fresh_mlp_target = candidate_rows.loc[
        (candidate_rows["evaluation_family"] == "fresh_mlp")
        & (candidate_rows["label"] == "target")
    ]
    if not (len(fixed_target) == len(fresh_linear_target) == len(fresh_mlp_target) == 60):
        raise RuntimeError("construct candidate endpoint rows are not exactly 60 per family")
    # The handoff did not predeclare a numerical or inferential materiality
    # threshold for fresh-decoder recovery. Keep those endpoints descriptive;
    # only the calibration-frozen inversion rule selects a branch.
    interpretation = (
        "inversion" if construct["inversion_count"] else "neither_failure"
    )
    linear_baseline_accuracy = float(
        construct["fresh_decoder_unedited_baselines"]["fresh_linear"]["target"]["metrics"]["accuracy"]
    )
    mlp_baseline_accuracy = float(
        construct["fresh_decoder_unedited_baselines"]["fresh_mlp"]["target"]["metrics"]["accuracy"]
    )
    construct["fresh_decoder_claim"] = {
        "status": "descriptive_only",
        "reason": "no predeclared materiality threshold or inferential test",
        "redecodability_claim_made": False,
        "count_above_chance_descriptive": int(construct["redecodable_count"]),
    }
    construct["manuscript_values"] = {
        "cell": "Qwen2.5-1.5B / SST-2 / layer 14",
        "candidate_edits": 60,
        "raw_median_damage": f"{float(fixed_target['C_raw'].median()):.3f}",
        "orientation_median_damage": f"{float(fixed_target['C_orientation_calibrated'].median()):.3f}",
        "fresh_linear_median_accuracy": f"{float(fresh_linear_target['accuracy'].median()):.3f}",
        "fresh_mlp_median_accuracy": f"{float(fresh_mlp_target['accuracy'].median()):.3f}",
        "fresh_linear_unedited_baseline_accuracy": f"{linear_baseline_accuracy:.3f}",
        "fresh_mlp_unedited_baseline_accuracy": f"{mlp_baseline_accuracy:.3f}",
        "fresh_decoder_inference": "descriptive_only",
        "interpretation": interpretation,
    }
    epsilon.update(
        {
            "bootstrap_seed": int(config.raw["reproducibility"]["bootstrap_seed"]),
            "bootstrap_draws": int(config.raw["reproducibility"]["bootstrap_draws"]),
            "permutation_seed": int(config.raw["reproducibility"]["permutation_seed"]),
        }
    )
    analysis_artifact_hashes = {
        filename: sha256_file(context.run_dir / filename)
        for filename in (
            "matched_split_rows.parquet",
            "matched_split_rows.csv",
            "epsilon_sweep_rows.parquet",
            "epsilon_sweep_rows.csv",
            "construct_check_rows.parquet",
            "construct_check_rows.csv",
        )
    }
    analysis_artifact_hashes.update(construct["referenced_artifact_hashes"])
    analysis = {
        "schema_version": 1,
        "status": "complete",
        "validation": {
            "complete": True,
            "exact_key_coverage": True,
            "zero_failed_units": True,
            "epsilon_zero_integrity": True,
            "baseline_ceiling_reproduced": True,
        },
        "run_id": context.run_id,
        "git_commit": _current_commit(),
        "config_hash": config.config_hash,
        "matched_split": matched,
        "epsilon_sweep": epsilon,
        "construct_check": construct,
        "artifact_hashes": analysis_artifact_hashes,
        "scientific_interpretation_limits": [
            "Fixed-decoder target damage is not evidence of erasure.",
            "Layers are nested in 12 model-task blocks and are not independent datasets.",
            "Orientation and fresh-decoder endpoints cover one prespecified Qwen/SST-2/layer-14 cell.",
        ],
    }
    atomic_write_json(context.run_dir / "analysis_summary.json", analysis)
    validation_lines = [
        "# Reviewer revision validation report",
        "",
        f"- Run: `{context.run_id}`",
        f"- Commit: `{analysis['git_commit']}`",
        f"- Config SHA-256: `{config.config_hash}`",
        (
            f"- Matched/split: {matched['included_units']['cells']} cells nested in "
            f"12 blocks; {matched['included_units']['pairs']} pair units; zero failed units."
        ),
        (
            f"- Epsilon sweep: {epsilon['required_middle_rows']} required middle-layer "
            "rows; epsilon-zero integrity passed; archived-budget ceiling reproduced."
        ),
        (
            f"- Construct check: {construct['included_units']['edits']} edits and "
            f"{construct['included_units']['rows']} endpoint rows in the prespecified cell."
        ),
        "- Exact keys, cache hashes, checkpoint hashes, edit hashes, split manifests, and per-example outputs are persisted.",
        "- Manuscript patching remains gated on figure generation and this validated analysis.",
        "",
        "Fixed-decoder target damage is not evidence of erasure.",
    ]
    report_text = "\n".join(validation_lines) + "\n"
    atomic_write_via(
        context.run_dir / "validation_report.md",
        lambda temporary: temporary.write_text(report_text, encoding="utf-8", newline="\n"),
    )
    context.update_manifest({"analysis": {"status": "ok"}})
    return analysis


def run_figures(context: RunContext, config: RevisionConfig) -> dict[str, Any]:
    analysis_path = context.run_dir / "analysis_summary.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if analysis.get("status") != "complete":
        raise RuntimeError("figures require a complete analysis summary")
    from .figures import generate_revision_figures

    artifacts = generate_revision_figures(
        context.run_dir,
        output_directory=context.run_dir / "figures",
        bootstrap_draws=int(config.raw["reproducibility"]["bootstrap_draws"]),
        bootstrap_seed=int(config.raw["reproducibility"]["bootstrap_seed"]),
    )
    project_figure_dir = PROJECT_ROOT / "figures"
    project_figure_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"status": "ok", "figures": {}}
    for name, artifact in artifacts.items():
        project_pdf = project_figure_dir / artifact.pdf_path.name
        project_png = project_figure_dir / artifact.png_path.name
        shutil.copyfile(artifact.pdf_path, project_pdf)
        shutil.copyfile(artifact.png_path, project_png)
        report["figures"][name] = {
            "stem": artifact.stem,
            "run_pdf": _relative(artifact.pdf_path, context.run_dir),
            "run_png": _relative(artifact.png_path, context.run_dir),
            "pdf_sha256": sha256_file(artifact.pdf_path),
            "png_sha256": sha256_file(artifact.png_path),
            "metadata": dict(artifact.metadata),
        }
    atomic_write_json(context.run_dir / "figures_report.json", report)
    context.update_manifest({"figures": {"status": "ok", "count": len(artifacts)}})
    return report


def _copy_pipeline_figure(context: RunContext) -> dict[str, Any]:
    source = PROJECT_ROOT / "assets" / "reviewer_revision" / "fig_pipeline.pdf"
    provenance_path = source.with_suffix(".provenance.json")
    if not source.is_file():
        raise FileNotFoundError(
            "the vendored preserved vector pipeline figure is missing"
        )
    if not provenance_path.is_file():
        raise FileNotFoundError("the vendored pipeline figure provenance is missing")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("sha256") != sha256_file(source):
        raise RuntimeError("the vendored pipeline figure hash differs from provenance")
    destinations = [
        PROJECT_ROOT / "figures" / "fig_pipeline.pdf",
        context.run_dir / "figures" / "fig_pipeline.pdf",
    ]
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return {
        "source_ref": _relative(source, PROJECT_ROOT),
        "source_sha256": sha256_file(source),
        "provenance_ref": _relative(provenance_path, PROJECT_ROOT),
        "provenance_sha256": sha256_file(provenance_path),
        "project_sha256": sha256_file(destinations[0]),
        "run_sha256": sha256_file(destinations[1]),
    }


def _main_text_page_count_from_text(page_texts: Iterable[str]) -> int:
    for page_number, page_text in enumerate(page_texts, start=1):
        if re.search(r"(?im)^\s*References\s*$", page_text):
            return page_number - 1
    raise ValueError("compiled manuscript has no identifiable References page")


def _pdfinfo_author_is_anonymous(info: str) -> bool:
    """Accept absent, blank, or explicitly anonymous PDF author metadata."""

    match = re.search(r"^Author:[ \t]*(.*)$", info, flags=re.MULTILINE)
    if match is None:
        return True
    return match.group(1).strip().casefold() in {
        "",
        "anonymous author(s)",
        "anonymous authors",
    }


def _decode_pdftotext_output(output: bytes) -> str:
    """Decode Poppler text deterministically across Windows code pages."""

    return output.decode("utf-8", errors="replace")


def _compile_manuscript(context: RunContext) -> dict[str, Any]:
    command = [
        "latexmk",
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "main_revised.tex",
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=1800,
    )
    elapsed = time.perf_counter() - started
    output = completed.stdout + "\n" + completed.stderr
    log_path = PROJECT_ROOT / "main_revised.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    undefined_patterns = (
        "There were undefined references",
        "Citation `",
        "undefined citations",
        "Reference `",
    )
    undefined = [pattern for pattern in undefined_patterns if pattern in log_text]
    overfull = [
        float(value)
        for value in re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", log_text)
    ]
    if completed.returncode != 0:
        raise RuntimeError(f"latexmk failed:\n{output[-8000:]}")
    if undefined:
        raise RuntimeError(f"manuscript has undefined citations/references: {undefined}")
    if overfull and max(overfull) > 3.0:
        raise RuntimeError(f"manuscript has overfull hbox above 3pt: {max(overfull):.3f}pt")
    pdf_path = PROJECT_ROOT / "main_revised.pdf"
    if not pdf_path.is_file():
        raise RuntimeError("latexmk returned success without main_revised.pdf")
    paper_dir = context.run_dir / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pdf_path, paper_dir / "main_revised.pdf")
    atomic_write_via(
        paper_dir / "latexmk_output.txt",
        lambda temporary: temporary.write_text(output, encoding="utf-8", newline="\n"),
    )
    render_dir = paper_dir / "rendered_pages"
    render_dir.mkdir(parents=True, exist_ok=True)
    render = subprocess.run(
        ["pdftoppm", "-png", "-r", "120", str(pdf_path), str(render_dir / "page")],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=600,
    )
    if render.returncode != 0:
        raise RuntimeError(f"PDF page rendering failed: {render.stderr}")
    info = subprocess.run(
        ["pdfinfo", str(pdf_path)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    ).stdout
    if not _pdfinfo_author_is_anonymous(info):
        raise RuntimeError("compiled PDF metadata contains a non-anonymous author")
    pages_match = re.search(r"^Pages:\s*(\d+)", info, flags=re.MULTILINE)
    page_count = int(pages_match.group(1)) if pages_match else None
    rendered_pages = sorted(render_dir.glob("page-*.png"))
    if page_count is None or len(rendered_pages) != page_count:
        raise RuntimeError("rendered page count does not match compiled PDF")
    page_texts: list[str] = []
    for page_number in range(1, page_count + 1):
        extraction = subprocess.run(
            [
                "pdftotext",
                "-f",
                str(page_number),
                "-l",
                str(page_number),
                str(pdf_path),
                "-",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
            timeout=60,
        )
        if extraction.returncode != 0:
            raise RuntimeError(
                f"could not extract PDF page {page_number}: "
                f"{_decode_pdftotext_output(extraction.stderr)}"
            )
        page_texts.append(_decode_pdftotext_output(extraction.stdout))
    main_text_pages = _main_text_page_count_from_text(page_texts)
    workshop_main_text_limit = 5
    if main_text_pages > workshop_main_text_limit:
        raise RuntimeError(
            "workshop main-text page limit exceeded: "
            f"{main_text_pages} > {workshop_main_text_limit}"
        )
    return {
        "status": "compiled",
        "command": " ".join(command),
        "wall_seconds": elapsed,
        "pdf_sha256": sha256_file(pdf_path),
        "pages": page_count,
        "main_text_pages": main_text_pages,
        "workshop_main_text_page_limit": workshop_main_text_limit,
        "workshop_limit_source": (
            "https://interpretability4discovery.github.io/cfp.html"
        ),
        "maximum_overfull_hbox_pt": max(overfull) if overfull else 0.0,
        "undefined_references_or_citations": False,
        "pdfinfo": info,
        "rendered_pages": [
            _relative(path, context.run_dir) for path in rendered_pages
        ],
        "visual_inspection": "pending",
    }


def run_patch_paper(context: RunContext, config: RevisionConfig) -> dict[str, Any]:
    del config
    analysis_path = context.run_dir / "analysis_summary.json"
    figures_report = _require_ok_report(context, "figures_report.json")
    del figures_report
    from .paper import generate_manuscript_numbers, patch_manuscript

    source_snapshot = context.run_dir / "paper" / "main_revised.prepatch.tex"
    if not source_snapshot.is_file():
        if not MANUSCRIPT_TEMPLATE_PATH.is_file():
            raise RuntimeError(
                f"missing locked manuscript template: {MANUSCRIPT_TEMPLATE_PATH}"
            )
        source_text = MANUSCRIPT_TEMPLATE_PATH.read_text(encoding="utf-8")
        template_text_sha256 = hashlib.sha256(source_text.encode()).hexdigest()
        if template_text_sha256 != MANUSCRIPT_TEMPLATE_TEXT_SHA256:
            raise RuntimeError(
                "locked manuscript template text hash mismatch: "
                f"expected {MANUSCRIPT_TEMPLATE_TEXT_SHA256}, "
                f"got {template_text_sha256}"
            )
        atomic_write_via(
            source_snapshot,
            lambda temporary: temporary.write_text(source_text, encoding="utf-8", newline="\n"),
        )
    run_macros = context.run_dir / "manuscript_numbers.tex"
    generate_manuscript_numbers(analysis_path, run_macros)
    shutil.copyfile(run_macros, PROJECT_ROOT / "manuscript_numbers.tex")
    patch_report = patch_manuscript(
        source_snapshot,
        PROJECT_ROOT / "main_revised.tex",
        analysis_path,
        macros_path=run_macros,
    )
    pipeline = _copy_pipeline_figure(context)
    compilation = _compile_manuscript(context)
    report = {
        "status": "compiled_pending_visual_inspection",
        "patch": {
            "destination": _relative(patch_report.destination, PROJECT_ROOT),
            "macros_path": _relative(run_macros, context.run_dir),
            "changed_regions": list(patch_report.changed_regions),
        },
        "pipeline_figure": pipeline,
        "compilation": compilation,
    }
    atomic_write_json(context.run_dir / "paper_report.json", report)
    context.update_manifest(
        {"paper": {"status": "compiled_pending_visual_inspection", "pages": compilation["pages"]}}
    )
    return report


_STAGE_SEQUENCE = (
    "preflight",
    "benchmark",
    "reproduce-baseline",
    "matched-split",
    "epsilon-sweep",
    "construct-check",
    "analyze",
    "figures",
    "patch-paper",
)

_STAGE_REPORTS = {
    "preflight": ("preflight_report.json", {"ok"}),
    "benchmark": ("runtime_benchmark.json", {"ok"}),
    "reproduce-baseline": ("baseline_reproduction.json", {"ok"}),
    "matched-split": ("matched_split_summary.json", {"ok"}),
    "epsilon-sweep": ("epsilon_sweep_summary.json", {"ok"}),
    "construct-check": ("construct_check_summary.json", {"ok"}),
    "analyze": ("analysis_summary.json", {"complete"}),
    "figures": ("figures_report.json", {"ok"}),
    "patch-paper": (
        "paper_report.json",
        {"compiled_pending_visual_inspection", "visually_inspected"},
    ),
}


def _select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    if requested == "mps":
        if not mps_available:
            raise RuntimeError("--device mps was requested, but MPS is unavailable")
        return torch.device("mps")
    if requested != "auto":
        raise ValueError(f"unsupported reviewer-revision device: {requested!r}")
    return torch.device("mps" if mps_available else "cpu")


def _configure_reproducibility(config: RevisionConfig) -> None:
    for name, value in config.raw["runtime"]["env"].items():
        os.environ[str(name)] = str(value)
    seed = int(config.raw["reproducibility"]["master_seed"])
    set_seed(seed)
    torch.use_deterministic_algorithms(
        bool(config.raw["reproducibility"]["deterministic_algorithms"])
    )


def _run_relative_path(context: RunContext, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise RuntimeError("artifact reference is not a nonempty relative path")
    path = (context.run_dir / relative).resolve()
    try:
        path.relative_to(context.run_dir.resolve())
    except ValueError as exc:
        raise RuntimeError(f"artifact reference escapes the run directory: {relative}") from exc
    return path


def _require_referenced_files(context: RunContext, references: Iterable[Any]) -> None:
    for reference in sorted({str(value) for value in references if value is not None}):
        path = _run_relative_path(context, reference.split("#", 1)[0])
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"referenced artifact is missing or empty: {reference}")


def _stage_is_complete(
    context: RunContext,
    stage: str,
    *,
    config: RevisionConfig,
) -> bool:
    filename, accepted_statuses = _STAGE_REPORTS[stage]
    path = context.run_dir / filename
    if not path.is_file():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if report.get("status") not in accepted_statuses:
        return False
    if stage == "preflight":
        environment_path = context.run_dir / "environment.json"
        if not environment_path.is_file() or not report.get("scientific_gates", {}).get(
            "all_intervention_layers_present"
        ):
            return False
    if stage == "benchmark":
        if report.get("selected_grid") not in {"full", "fallback"}:
            return False
        if report.get("epsilon_scope") not in {"middle_only", "all_selected_layers"}:
            return False
    if stage == "reproduce-baseline" and report.get("archive_sha256") != sha256_file(
        ARCHIVE_PATH
    ):
        raise RuntimeError("archived baseline changed after its completed gate")
    if stage == "reproduce-baseline":
        current_ref = report.get("current_rerun_ref")
        current_hash = report.get("current_rerun_sha256")
        if not isinstance(current_ref, str) or not isinstance(current_hash, str):
            return False
        current_path = _run_relative_path(context, current_ref)
        if not current_path.is_file() or sha256_file(current_path) != current_hash:
            raise RuntimeError("current-code baseline rerun artifact hash mismatch")
    if stage in {"matched-split", "epsilon-sweep", "construct-check"}:
        stem = {
            "matched-split": "matched_split",
            "epsilon-sweep": "epsilon_sweep",
            "construct-check": "construct_check",
        }[stage]
        parquet_path = context.run_dir / f"{stem}_rows.parquet"
        csv_path = context.run_dir / f"{stem}_rows.csv"
        if not parquet_path.is_file() or not csv_path.is_file():
            raise RuntimeError(f"completed {stage} row artifacts are missing")
        if report.get("raw_row_file_sha256") != sha256_file(parquet_path):
            raise RuntimeError(f"completed {stage} parquet hash mismatch")
        frame = pd.read_parquet(parquet_path)
        if stage == "matched-split":
            benchmark = _load_benchmark_report(context)
            expected = config.matched_split_row_keys(benchmark["selected_grid"])
            if context.completed_keys("matched_split", expected_keys=expected) != {
                tuple(key) for key in expected
            }:
                raise RuntimeError("completed matched/split shard set is incomplete")
            for column in (
                "split_manifest_ref",
                "edit_artifact_ref",
                "per_example_artifact_ref",
            ):
                if column not in frame or frame[column].isna().any():
                    raise RuntimeError(
                        f"completed matched/split rows lack required {column} references"
                    )
                _require_referenced_files(context, frame[column].tolist())
            coupled = frame.loc[frame["condition"].isin(("matched", "split"))]
            for _, pair_rows in coupled.groupby(
                ["model_key", "task", "layer", "pair_seed", "method"],
                dropna=False,
            ):
                if len(pair_rows) != 2:
                    raise RuntimeError("coupled matched/split artifact pair is incomplete")
                if pair_rows["edit_hash"].nunique() != 1:
                    raise RuntimeError("coupled matched/split rows have different edit hashes")
                if pair_rows["edit_artifact_ref"].nunique() != 1:
                    raise RuntimeError("coupled matched/split rows reference different edits")
            for _, row in frame.drop_duplicates(
                ["model_key", "task", "layer", "pair_seed"]
            ).iterrows():
                checkpoint = _checkpoint_path(
                    context,
                    model_key=str(row["model_key"]),
                    task=str(row["task"]),
                    layer=int(row["layer"]),
                    pair_seed=int(row["pair_seed"]),
                )
                if not checkpoint.is_file():
                    raise RuntimeError(f"completed pair checkpoint is missing: {checkpoint.name}")
        elif stage == "epsilon-sweep":
            benchmark = _load_benchmark_report(context)
            expected = _epsilon_expected_keys(
                config,
                scope=benchmark["epsilon_scope"],
                grid=benchmark["selected_grid"],
            )
            if context.completed_keys("epsilon_sweep", expected_keys=expected) != {
                tuple(key) for key in expected
            }:
                raise RuntimeError("completed epsilon shard set is incomplete")
            for column in (
                "split_manifest_ref",
                "edit_artifact_ref",
                "per_example_artifact_ref",
            ):
                _require_referenced_files(context, frame[column].tolist())
        else:
            expected_edits = config.construct_edit_keys()
            if context.completed_keys(
                "construct_edits", expected_keys=expected_edits
            ) != {tuple(key) for key in expected_edits}:
                raise RuntimeError("completed construct shard set is incomplete")
            _require_referenced_files(context, frame["split_manifest_ref"].tolist())
            for column in ("per_example_refs", "training_curve_refs"):
                references: list[str] = []
                if column in frame:
                    for value in frame[column].dropna():
                        if isinstance(value, (list, tuple, np.ndarray)):
                            references.extend(str(item) for item in value)
                _require_referenced_files(context, references)
    if stage == "analyze" and report.get("validation", {}).get("complete") is not True:
        return False
    if stage == "analyze":
        for relative, expected_hash in report.get("artifact_hashes", {}).items():
            path = _run_relative_path(context, relative)
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise RuntimeError(f"analysis input hash mismatch: {relative}")
        from .paper import validate_analysis_summary

        validate_analysis_summary(report)
    if stage == "figures":
        figures = report.get("figures")
        if not isinstance(figures, dict) or not figures:
            return False
        for artifact in figures.values():
            for key in ("run_pdf", "run_png"):
                relative = artifact.get(key)
                if not isinstance(relative, str):
                    return False
                artifact_path = _run_relative_path(context, relative)
                expected_hash = artifact.get("pdf_sha256" if key == "run_pdf" else "png_sha256")
                if not artifact_path.is_file() or sha256_file(artifact_path) != expected_hash:
                    raise RuntimeError(f"completed figure artifact hash mismatch: {relative}")
    if stage == "patch-paper":
        compilation = report.get("compilation", {})
        if compilation.get("status") != "compiled":
            return False
        if not (context.run_dir / "paper" / "main_revised.pdf").is_file():
            return False
        if sha256_file(context.run_dir / "paper" / "main_revised.pdf") != compilation.get(
            "pdf_sha256"
        ):
            raise RuntimeError("compiled paper PDF hash mismatch")
    return True


def _validate_stage_result(stage: str, result: Mapping[str, Any]) -> None:
    accepted = _STAGE_REPORTS[stage][1]
    status = result.get("status")
    if status not in accepted:
        raise RuntimeError(
            f"{stage} returned status {status!r}; expected one of {sorted(accepted)!r}"
        )
    if stage == "analyze" and result.get("validation", {}).get("complete") is not True:
        raise RuntimeError("analysis did not pass its complete validation gate")


def _run_stage(
    stage: str,
    context: RunContext,
    config: RevisionConfig,
    *,
    device: torch.device,
) -> Mapping[str, Any]:
    if stage == "preflight":
        return run_preflight(context, config, device=device)
    if stage == "benchmark":
        return run_benchmark(context, config, device=device)
    if stage == "reproduce-baseline":
        return run_reproduce_baseline(context, config)
    if stage == "matched-split":
        return run_matched_split(context, config, device=device)
    if stage == "epsilon-sweep":
        return run_epsilon_sweep(context, config, device=device)
    if stage == "construct-check":
        return run_construct_check(context, config, device=device)
    if stage == "analyze":
        return run_analysis(context, config)
    if stage == "figures":
        return run_figures(context, config)
    if stage == "patch-paper":
        return run_patch_paper(context, config)
    raise ValueError(f"unknown reviewer-revision stage: {stage!r}")


def execute(command: str, config: RevisionConfig, args: Any) -> int:
    """Execute one locked stage, or the complete fail-closed reviewer pipeline."""

    if command != "all" and command not in _STAGE_SEQUENCE:
        raise ValueError(f"unknown reviewer-revision command: {command!r}")
    _configure_reproducibility(config)
    device = _select_device(str(args.device))
    git_commit = _current_commit()
    output_root = Path(args.output_root)
    if bool(args.resume):
        run_dir = resolve_resume_directory(output_root, config_hash=config.config_hash)
        context = RunContext.resume(
            run_dir,
            config_hash=config.config_hash,
            git_commit=git_commit,
        )
    else:
        context = RunContext.create(
            output_root=output_root,
            config_hash=config.config_hash,
            git_commit=git_commit,
            manifest={
                "starting_commit": _starting_commit(),
                "branch": _git("branch", "--show-current") or "DETACHED",
                "handoff_inputs": _input_manifest(),
                "config_source": Path(args.config).name,
                "requested_device": str(args.device),
                "selected_device": str(device),
            },
        )

    file_handler = logging.FileHandler(context.console_log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    tee_handle = context.console_log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(original_stdout, tee_handle)
    sys.stderr = _TeeStream(original_stderr, tee_handle)
    stages = _STAGE_SEQUENCE if command == "all" else (command,)
    completed: list[str] = []
    skipped: list[str] = []
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        with context:
            context.update_manifest(
                {
                    "pipeline": {
                        "status": "running",
                        "command": command,
                        "started_at_utc": started_at,
                    }
                }
            )
            LOGGER.info(
                "run %s: command=%s device=%s resume=%s",
                context.run_id,
                command,
                device,
                bool(args.resume),
            )
            for stage in stages:
                if bool(args.resume) and _stage_is_complete(
                    context, stage, config=config
                ):
                    LOGGER.info("%s: validated completed report; skipping", stage)
                    skipped.append(stage)
                    continue
                LOGGER.info("%s: starting", stage)
                result = _run_stage(stage, context, config, device=device)
                _validate_stage_result(stage, result)
                completed.append(stage)
                context.update_manifest(
                    {
                        "pipeline": {
                            "status": "running",
                            "command": command,
                            "started_at_utc": started_at,
                            "last_completed_stage": stage,
                            "completed_this_invocation": completed,
                            "skipped_validated": skipped,
                        }
                    }
                )
                LOGGER.info("%s: completed", stage)

            final_stage = stages[-1]
            final_status = (
                "pending_visual_inspection"
                if final_stage == "patch-paper"
                else "stage_complete"
            )
            context.update_manifest(
                {
                    "pipeline": {
                        "status": final_status,
                        "command": command,
                        "started_at_utc": started_at,
                        "finished_at_utc": datetime.now(timezone.utc)
                        .isoformat()
                        .replace("+00:00", "Z"),
                        "last_completed_stage": final_stage,
                        "completed_this_invocation": completed,
                        "skipped_validated": skipped,
                    }
                }
            )
            LOGGER.info("run %s reached %s", context.run_id, final_status)
        return 0
    except BaseException as exc:
        LOGGER.exception("reviewer-revision pipeline failed")
        try:
            manifest = json.loads(context.manifest_path.read_text(encoding="utf-8"))
            manifest["pipeline"] = sanitize_manifest_payload(
                {
                    "status": "failed",
                    "command": command,
                    "started_at_utc": started_at,
                    "failed_at_utc": datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "last_completed_stage": completed[-1] if completed else None,
                    "completed_this_invocation": completed,
                    "skipped_validated": skipped,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            atomic_write_json(context.manifest_path, manifest)
        except Exception:
            LOGGER.exception("could not persist the pipeline failure manifest")
        finally:
            context.close()
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        tee_handle.close()
        root_logger.removeHandler(file_handler)
        file_handler.close()
