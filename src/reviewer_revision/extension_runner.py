"""Fail-closed orchestration for the prospectively locked caveat extension."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import psutil
import torch

from .artifacts import (
    RunContext,
    assert_exact_keys,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_via,
    sanitize_manifest_payload,
    sha256_file,
)
from .config import RevisionConfig
from .extension_config import ConstructCell, ExtensionConfig

LOGGER = logging.getLogger("reviewer_revision.extension")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "reviewer_caveat_extension_2026_08"

STAGES = (
    "preflight",
    "robustness",
    "construct-panel",
    "analyze",
    "package-artifacts",
    "figures",
    "patch-paper",
)

_STAGE_REPORTS: dict[str, tuple[str, frozenset[str]]] = {
    "preflight": ("preflight_report.json", frozenset(("ok",))),
    "robustness": ("floor_robustness_summary.json", frozenset(("ok",))),
    "construct-panel": ("construct_panel_report.json", frozenset(("ok",))),
    "analyze": ("extension_analysis_summary.json", frozenset(("complete",))),
    "package-artifacts": ("portability_report.json", frozenset(("ok",))),
    "figures": ("extension_figures_report.json", frozenset(("ok",))),
    "patch-paper": (
        "paper_extension_report.json",
        frozenset(("compiled_pending_visual_inspection",)),
    ),
}

_BASE_INPUTS = (
    "run_manifest.json",
    "artifact_validation_report.json",
    "matched_split_rows.parquet",
    "matched_split_summary.json",
    "construct_check_rows.parquet",
    "construct_check_summary.json",
    "analysis_summary.json",
)
_CACHE_TAGS = ("cand", "eval", "inter")
_CONSTRUCT_CELL_EXPERIMENT = "construct-panel-cells"
_GROUP_ROWS_PARQUET = "construct_panel_group_rows.parquet"
_GROUP_ROWS_CSV = "construct_panel_group_rows.csv"
_ANALYSIS_SUMMARY = "extension_analysis_summary.json"
_REGISTERED_BASE_COMMIT = "7aab3eb9f3145da17c96ea05020353eef48904a4"


class _KnownPaddingCacheError(RuntimeError):
    """Internal sentinel for the only cache condition authorized for repair."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def current_git_commit() -> str:
    """Return the exact source commit bound into a new or resumed run."""

    commit = _git("rev-parse", "HEAD")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError("git rev-parse did not return a full lowercase commit")
    return commit


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _run_path(context: RunContext, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise RuntimeError("artifact reference must be a nonempty relative path")
    candidate = (context.run_dir / relative).resolve()
    if not candidate.is_relative_to(context.run_dir.resolve()):
        raise RuntimeError(f"artifact reference escapes run directory: {relative!r}")
    return candidate


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not read JSON artifact {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"JSON artifact {path.name} must contain an object")
    return payload


def _atomic_copy(source: Path, destination: Path) -> Path:
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"source artifact is missing or empty: {source}")
    return atomic_write_via(destination, lambda temporary: shutil.copyfile(source, temporary))


def build_dry_run_plan(
    extension: ExtensionConfig,
    base: RevisionConfig,
) -> dict[str, Any]:
    """Describe exact work without touching caches, run directories, or endpoints."""

    extension.validate()
    edit_keys = base.construct_edit_keys()
    candidate_edits = tuple(key for key in edit_keys if key.edit_kind == "dcand_crossfit")
    confirmatory_cells = len(extension.confirmatory_cells)
    return {
        "confirmatory_cells": confirmatory_cells,
        "pilot_cells": 1,
        "worker_edits_per_cell": len(edit_keys),
        "total_worker_edits": confirmatory_cells * len(edit_keys),
        "inferential_candidate_edits": confirmatory_cells * len(candidate_edits),
        "compatibility_rows": confirmatory_cells * len(edit_keys) * 3 * 2,
        "bootstrap_draws": extension.bootstrap_draws,
        "bootstrap_seed": extension.bootstrap_seed,
        "stages": list(STAGES),
        "computes_confirmatory_endpoints": False,
    }


def require_construct_disk_capacity(
    *,
    free_bytes: int,
    projected_bytes: int,
    reserve_bytes: int,
) -> None:
    """Require projected new artifacts while retaining the locked reserve."""

    values = (free_bytes, projected_bytes, reserve_bytes)
    if any(type(value) is not int for value in values):
        raise TypeError("disk gate requires integer byte counts")
    if any(value < 0 for value in values):
        raise ValueError("disk gate byte counts must be nonnegative")
    required = projected_bytes + reserve_bytes
    if free_bytes < required:
        raise RuntimeError(
            "construct disk gate failed: "
            f"free={free_bytes}, projected={projected_bytes}, "
            f"reserve={reserve_bytes}, required={required}"
        )


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def select_extension_device(requested: str) -> torch.device:
    """Resolve CUDA, Apple MPS, or CPU without the legacy MPS-only restriction."""

    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "mps":
        if not _mps_available():
            raise RuntimeError("--device mps was requested, but MPS is unavailable")
        return torch.device("mps")
    if requested != "auto":
        raise ValueError(f"unsupported extension device: {requested!r}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_extension_resume_directory(
    output_root: str | Path,
    *,
    config_hash: str,
    git_commit: str,
) -> Path:
    """Find the newest unlocked run matching both immutable resume identities."""

    root = Path(output_root)
    candidates = (
        [root]
        if (root / "run_manifest.json").is_file()
        else sorted(
            (
                path
                for path in root.glob("*")
                if (path / "run_manifest.json").is_file()
            ),
            reverse=True,
        )
        if root.exists()
        else []
    )
    for candidate in candidates:
        try:
            manifest = _read_json(candidate / "run_manifest.json")
        except RuntimeError:
            continue
        if (
            manifest.get("schema_version") != 1
            or manifest.get("run_id") != candidate.name
            or manifest.get("config_hash") != config_hash
            or manifest.get("git_commit") != git_commit
        ):
            continue
        lock_path = candidate / ".run.lock"
        if lock_path.exists():
            try:
                lock_payload = _read_json(lock_path)
                owner_pid = int(lock_payload["pid"])
                alive = owner_pid > 0 and psutil.pid_exists(owner_pid)
            except (RuntimeError, TypeError, ValueError, KeyError):
                alive = False
            if alive:
                continue
            stale = candidate / f".stale-run-lock-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
            try:
                os.replace(lock_path, stale)
            except FileNotFoundError:
                pass
            except OSError:
                continue
        return candidate
    raise FileNotFoundError(
        f"no resumable run under {root} matches the exact config hash and Git commit"
    )


def planned_alterrep_keys(base: RevisionConfig) -> tuple[tuple[Any, ...], ...]:
    """Return the exact registered 600-row AlterRep matched/split universe."""

    return tuple(
        tuple(key)
        for key in base.matched_split_row_keys("full")
        if key.method == "alterrep" and key.condition in {"matched", "split"}
    )


# These wrappers are intentionally narrow dependency seams.  Tests can replace
# them without importing transformer/model code, while real runs stay on the
# validated scientific extraction and construct-worker implementations.
def reconstruct_tasks(config: RevisionConfig) -> Mapping[str, Any]:
    from .runner import _reconstruct_tasks

    return _reconstruct_tasks(config)


def load_construct_cache(**kwargs: Any) -> Any:
    from .runner import _load_cache

    return _load_cache(**kwargs)


def cache_requires_padding_fix(model_key: str, cache: Any) -> bool:
    from .runner import _requires_padding_fix_regeneration

    return _requires_padding_fix_regeneration(model_key, cache)


def regenerate_construct_cache_group(**kwargs: Any) -> dict[str, Any]:
    from .runner import _regenerate_cache_group

    return _regenerate_cache_group(**kwargs)


def run_construct_cell_worker(
    context: RunContext,
    config: RevisionConfig,
    *,
    cell: ConstructCell,
    device: torch.device,
) -> dict[str, Any]:
    from .runner import run_construct_cell

    return run_construct_cell(context, config, cell=cell, device=device)


def summarize_floor_robustness(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .extension_analysis import summarize_floor_robustness as implementation

    return implementation(*args, **kwargs)


def analyze_construct_panel(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .extension_analysis import classify_construct_panel

    return classify_construct_panel(*args, **kwargs)


def create_floor_sensitivity_figure(*args: Any, **kwargs: Any) -> Any:
    from .figures import create_floor_sensitivity_figure as implementation

    return implementation(*args, **kwargs)


def create_construct_panel_figure(*args: Any, **kwargs: Any) -> Any:
    from .figures import create_construct_panel_figure as implementation

    return implementation(*args, **kwargs)


def patch_manuscript_with_extension(*args: Any, **kwargs: Any) -> Any:
    from .paper import patch_manuscript_with_extension as implementation

    return implementation(*args, **kwargs)


def compile_workshop_manuscript(context: RunContext) -> dict[str, Any]:
    from .runner import _compile_manuscript

    return _compile_manuscript(context)


def _cache_record(cache: Any, cell: ConstructCell, tag: str) -> dict[str, Any]:
    selection = cache.selection
    path = Path(selection.path)
    if path.is_file() and sha256_file(path) != selection.cache_sha256:
        raise RuntimeError(f"validated cache bytes changed before pinning: {path.name}")
    record = {
        "model_key": cell.model_key,
        "model_id": cell.model_id,
        "task": cell.task,
        "layer": cell.layer,
        "tag": tag,
        "data_hash": selection.data_hash,
        "cache_sha256": selection.cache_sha256,
        "cache_path": _relative(path, PROJECT_ROOT),
        "n_examples": int(cache.n_examples),
        "hidden_size": int(cache.hidden_size),
        "mean_feature_variance": float(cache.mean_feature_variance),
        "class_conditioned_variance": cache.class_conditioned_variance,
        "legacy_dtype_semantics": bool(cache.legacy_dtype_semantics),
        "extraction_code_version": selection.provenance.get(
            "extraction_code_version"
        ),
    }
    metadata_path = path.with_suffix(".json")
    if metadata_path.is_file():
        record.update(
            {
                "metadata_path": _relative(metadata_path, PROJECT_ROOT),
                "metadata_sha256": sha256_file(metadata_path),
            }
        )
    return sanitize_manifest_payload(record)


def validate_construct_cache_pins(
    extension: ExtensionConfig,
    base: RevisionConfig,
    *,
    device: torch.device,
    cells: Sequence[ConstructCell] | None = None,
) -> dict[str, Any]:
    """Validate exact construct caches, repairing only identified Gemma padding caches."""

    requested = tuple(extension.all_cells if cells is None else cells)
    if not requested:
        raise ValueError("construct cache preflight requires at least one cell")
    reconstructions = reconstruct_tasks(base)
    cache_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    for cell in requested:
        if cell.task not in reconstructions:
            raise RuntimeError(f"missing reconstructed task for {cell.slug}")
        reconstruction = reconstructions[cell.task]

        def load_group(
            current_cell: ConstructCell = cell,
            current_reconstruction: Any = reconstruction,
        ) -> dict[str, Any]:
            loaded: dict[str, Any] = {}
            for tag in _CACHE_TAGS:
                cache = load_construct_cache(
                    model_id=current_cell.model_id,
                    task=current_cell.task,
                    layer=current_cell.layer,
                    tag=tag,
                    reconstruction=current_reconstruction,
                )
                if cache_requires_padding_fix(current_cell.model_key, cache):
                    raise _KnownPaddingCacheError(
                        f"{current_cell.slug}/{tag} predates corrected "
                        "last-nonpadding extraction"
                    )
                loaded[tag] = cache
            return loaded

        try:
            group = load_group()
        except _KnownPaddingCacheError as error:
            if cell.model_key != "gemma":
                raise RuntimeError(
                    "padding-cache repair was requested outside the known Gemma condition"
                ) from error
            recovery = regenerate_construct_cache_group(
                model=base.model(cell.model_key),
                task=cell.task,
                reconstruction=reconstruction,
                requirements=tuple((tag, cell.layer) for tag in _CACHE_TAGS),
                extraction_device=device,
                trigger=error,
            )
            group = load_group()
            recovery_rows.append(
                sanitize_manifest_payload(
                    {
                        **recovery,
                        "status": "regenerated_and_revalidated",
                        "validated_cache_sha256": {
                            tag: group[tag].selection.cache_sha256
                            for tag in _CACHE_TAGS
                        },
                    }
                )
            )
        for tag in _CACHE_TAGS:
            cache_rows.append(_cache_record(group[tag], cell, tag))

    expected = {
        (cell.model_id, cell.task, cell.layer, tag)
        for cell in requested
        for tag in _CACHE_TAGS
    }
    observed = {
        (row["model_id"], row["task"], row["layer"], row["tag"])
        for row in cache_rows
    }
    if observed != expected or len(cache_rows) != len(expected):
        raise RuntimeError("construct cache preflight did not produce exact cache pins")
    return {"caches": cache_rows, "cache_recovery": recovery_rows}


def _base_run_path(extension: ExtensionConfig) -> Path:
    path = (PROJECT_ROOT / extension.base_run).resolve()
    if not path.is_relative_to(PROJECT_ROOT.resolve()):
        raise RuntimeError("extension base run escapes the repository")
    return path


def base_provenance_mode(
    generating_commit: str,
    *,
    allow_reproduced_base: bool = False,
) -> str:
    """Classify a base run without weakening the registered default."""

    if (
        type(generating_commit) is not str
        or len(generating_commit) != 40
        or any(character not in "0123456789abcdef" for character in generating_commit)
    ):
        raise RuntimeError("base-run generating commit must be lowercase hexadecimal")
    if generating_commit == _REGISTERED_BASE_COMMIT:
        return "registered"
    if allow_reproduced_base:
        return "reproduced"
    raise RuntimeError("immutable base run generating commit differs from registration")


def validate_immutable_base_run(
    extension: ExtensionConfig,
    base: RevisionConfig,
    *,
    allow_reproduced_base: bool = False,
) -> dict[str, Any]:
    """Validate and content-pin every immutable base artifact used by the extension."""

    base_run = _base_run_path(extension)
    manifest = _read_json(base_run / "run_manifest.json")
    if manifest.get("run_id") != base_run.name:
        raise RuntimeError("immutable base run ID does not match its directory")
    if manifest.get("config_hash") != base.config_hash:
        raise RuntimeError("immutable base run config hash differs from the base config")
    provenance_mode = base_provenance_mode(
        manifest.get("git_commit"),
        allow_reproduced_base=allow_reproduced_base,
    )
    required_statuses = {
        "matched_split": "ok",
        "construct_check": "ok",
        "analysis": "ok",
    }
    for key, status in required_statuses.items():
        if manifest.get(key, {}).get("status") != status:
            raise RuntimeError(f"immutable base run lacks successful {key}")
    if manifest.get("finalization", {}).get("status") != "passed":
        raise RuntimeError("immutable base run did not pass finalization")

    records: list[dict[str, Any]] = []
    for relative in _BASE_INPUTS:
        path = base_run / relative
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"immutable base input is missing or empty: {relative}")
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    validation = _read_json(base_run / "artifact_validation_report.json")
    expected_rows_hash = validation.get("row_artifacts", {}).get(
        "matched_split_rows.parquet", {}
    ).get("sha256")
    observed_rows_hash = next(
        row["sha256"]
        for row in records
        if row["path"] == "matched_split_rows.parquet"
    )
    if validation.get("status") != "passed" or expected_rows_hash != observed_rows_hash:
        raise RuntimeError("immutable base validation report does not pin matched rows")
    return {
        "run_id": base_run.name,
        "config_hash": base.config_hash,
        "git_commit": manifest["git_commit"],
        "provenance_mode": provenance_mode,
        "inputs": records,
    }


def projected_construct_storage_bytes(
    base_run: str | Path,
    *,
    confirmatory_cells: int,
) -> dict[str, int]:
    """Project panel storage from the complete disclosed pilot footprint."""

    if type(confirmatory_cells) is not int or confirmatory_cells <= 0:
        raise ValueError("confirmatory_cells must be a positive integer")
    root = Path(base_run)
    candidates = (
        root / "construct",
        root / "checkpoints" / "construct",
        root / "shards" / "construct_edits",
    )
    files: set[Path] = set()
    for candidate in candidates:
        if candidate.is_dir():
            files.update(path.resolve() for path in candidate.rglob("*") if path.is_file())
    for filename in (
        "construct_check_rows.csv",
        "construct_check_rows.parquet",
        "construct_check_summary.json",
    ):
        path = root / filename
        if path.is_file():
            files.add(path.resolve())
    pilot_bytes = sum(path.stat().st_size for path in files)
    if pilot_bytes <= 0:
        raise RuntimeError("cannot project construct storage from an empty pilot")
    return {
        "pilot_artifact_bytes": pilot_bytes,
        "confirmatory_cells": confirmatory_cells,
        "projected_additional_bytes": pilot_bytes * confirmatory_cells,
    }


def run_preflight(
    context: RunContext,
    extension: ExtensionConfig,
    base: RevisionConfig,
    *,
    device: torch.device,
    allow_reproduced_base: bool = False,
) -> dict[str, Any]:
    base_record = validate_immutable_base_run(
        extension,
        base,
        allow_reproduced_base=allow_reproduced_base,
    )
    base_run = _base_run_path(extension)
    projection = projected_construct_storage_bytes(
        base_run,
        confirmatory_cells=len(extension.confirmatory_cells),
    )
    free_bytes = int(shutil.disk_usage(PROJECT_ROOT).free)
    reserve_bytes = int(extension.disk_reserve_gib * 2**30)
    require_construct_disk_capacity(
        free_bytes=free_bytes,
        projected_bytes=projection["projected_additional_bytes"],
        reserve_bytes=reserve_bytes,
    )
    cache_report = validate_construct_cache_pins(
        extension,
        base,
        device=device,
    )
    _atomic_copy(
        base_run / "matched_split_summary.json",
        context.run_dir / "matched_split_summary.json",
    )
    report = {
        "schema_version": 1,
        "status": "ok",
        "extension_config_hash": extension.config_hash,
        "base_config_hash": base.config_hash,
        "generating_git_commit": current_git_commit(),
        "immutable_base_run": base_record,
        "caches": cache_report["caches"],
        "cache_recovery": cache_report["cache_recovery"],
        "disk_gate": {
            **projection,
            "free_bytes": free_bytes,
            "reserve_bytes": reserve_bytes,
            "required_bytes": projection["projected_additional_bytes"]
            + reserve_bytes,
            "status": "passed",
        },
        "selected_device": str(device),
        "confirmatory_endpoints_computed": False,
    }
    atomic_write_json(context.run_dir / "preflight_report.json", report)
    context.update_manifest({"preflight": {"status": "ok"}})
    return report


def _verify_preflight_report(
    context: RunContext,
    extension: ExtensionConfig,
    base: RevisionConfig,
    report: Mapping[str, Any],
) -> None:
    if (
        report.get("status") != "ok"
        or report.get("extension_config_hash") != extension.config_hash
        or report.get("base_config_hash") != base.config_hash
        or report.get("generating_git_commit") != current_git_commit()
        or report.get("confirmatory_endpoints_computed") is not False
    ):
        raise RuntimeError("preflight report provenance or status is invalid")
    base_run = _base_run_path(extension)
    for record in report.get("immutable_base_run", {}).get("inputs", []):
        path = base_run / str(record.get("path"))
        if (
            not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or sha256_file(path) != record.get("sha256")
        ):
            raise RuntimeError(f"immutable base input changed: {record.get('path')}")
    cache_keys: list[tuple[str, str, int, str]] = []
    for record in report.get("caches", []):
        cache_keys.append(
            (
                str(record.get("model_id")),
                str(record.get("task")),
                int(record.get("layer")),
                str(record.get("tag")),
            )
        )
        path = (PROJECT_ROOT / str(record.get("cache_path"))).resolve()
        if not path.is_relative_to(PROJECT_ROOT.resolve()):
            raise RuntimeError("preflight cache reference escapes repository")
        if not path.is_file() or sha256_file(path) != record.get("cache_sha256"):
            raise RuntimeError(f"preflight cache pin changed: {record.get('cache_path')}")
        metadata_reference = record.get("metadata_path")
        metadata_digest = record.get("metadata_sha256")
        if not isinstance(metadata_reference, str) or not isinstance(
            metadata_digest, str
        ):
            raise TypeError("preflight cache pin lacks provenance sidecar hash")
        metadata_path = (PROJECT_ROOT / metadata_reference).resolve()
        if (
            not metadata_path.is_relative_to(PROJECT_ROOT.resolve())
            or not metadata_path.is_file()
            or sha256_file(metadata_path) != metadata_digest
        ):
            raise RuntimeError(
                f"preflight cache provenance changed: {metadata_reference}"
            )
    expected = [
        (cell.model_id, cell.task, cell.layer, tag)
        for cell in extension.all_cells
        for tag in _CACHE_TAGS
    ]
    assert_exact_keys(expected, cache_keys)
    summary_copy = context.run_dir / "matched_split_summary.json"
    source_summary = base_run / "matched_split_summary.json"
    if not summary_copy.is_file() or sha256_file(summary_copy) != sha256_file(source_summary):
        raise RuntimeError("immutable matched/split summary copy changed")


def run_robustness(
    context: RunContext,
    extension: ExtensionConfig,
    base: RevisionConfig,
) -> dict[str, Any]:
    preflight = _read_json(context.run_dir / "preflight_report.json")
    _verify_preflight_report(context, extension, base, preflight)
    rows_path = _base_run_path(extension) / "matched_split_rows.parquet"
    summary = summarize_floor_robustness(
        rows_path,
        expected_alterrep_keys=planned_alterrep_keys(base),
        draws=extension.bootstrap_draws,
        seed=extension.bootstrap_seed,
    )
    if summary.get("status") != "ok":
        raise RuntimeError("floor robustness did not return status ok")
    result = {
        **summary,
        "extension_config_hash": extension.config_hash,
        "base_config_hash": base.config_hash,
        "generating_git_commit": current_git_commit(),
        "source_rows_sha256": sha256_file(rows_path),
    }
    atomic_write_json(context.run_dir / "floor_robustness_summary.json", result)
    context.update_manifest({"robustness": {"status": "ok"}})
    return result


def _cell_group_paths(run_dir: Path, cell: ConstructCell) -> tuple[Path, Path]:
    root = run_dir / "construct" / "cells" / cell.slug
    return root / "construct_group_rows.parquet", root / "construct_group_rows.csv"


def _load_generic_group_rows(run_dir: Path, cell: ConstructCell) -> pd.DataFrame:
    parquet, _ = _cell_group_paths(run_dir, cell)
    if not parquet.is_file() or parquet.stat().st_size <= 0:
        raise RuntimeError(f"construct worker did not emit group rows for {cell.slug}")
    return pd.read_parquet(parquet)


def build_pilot_group_rows(
    base_run: Path,
    base: RevisionConfig,
    cell: ConstructCell,
) -> pd.DataFrame:
    """Reconstruct disclosed pilot group counts from immutable legacy artifacts."""

    from .runner import build_construct_group_rows

    read_only_context = SimpleNamespace(
        run_dir=base_run,
        run_id=base_run.name,
    )
    return build_construct_group_rows(
        read_only_context,
        base,
        cell=cell,
        legacy_layout=True,
    )


def validate_construct_group_rows_without_inference(
    rows: pd.DataFrame | Path | str | Iterable[Mapping[str, Any]],
    config: ExtensionConfig,
) -> dict[str, Any]:
    """Validate all exact row keys without calculating any scientific endpoint."""

    from .extension_analysis import (
        _prepare_construct_cell,
        _validate_construct_group_rows,
    )

    frame, blocked = _validate_construct_group_rows(rows, config=config)
    prepared: list[str] = []
    for cell in config.all_cells:
        if cell.slug in blocked:
            continue
        _prepare_construct_cell(frame, cell)
        prepared.append(cell.slug)
    return {
        "cell_count": len(config.all_cells),
        "validated_estimable_cells": prepared,
        "blocked": blocked,
        "confirmatory_endpoints_computed": False,
    }


def _cell_status_frame(cell: ConstructCell, error: BaseException) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_key": cell.model_key,
                "model_id": cell.model_id,
                "task": cell.task,
                "layer": cell.layer,
                "row_kind": "cell_status",
                "evaluation_family": "fresh_linear",
                "decoder_seed": 0,
                "label": None,
                "group_id": None,
                "n_examples": 0,
                "correct_count": 0,
                "split_manifest_sha256": None,
                "decoder_checkpoint_sha256": None,
                "edit_id": None,
                "edit_object": None,
                "architecture": None,
                "candidate_seed": None,
                "edit_hash": None,
                "status": "failed",
                "failure_stage": "construct_cell",
                "failure_reason": f"{type(error).__name__}: {error}",
            }
        ]
    )


def _validate_completed_cell_shard(
    context: RunContext,
    cell: ConstructCell,
) -> None:
    payload = context.load_json_shard(
        _CONSTRUCT_CELL_EXPERIMENT,
        (cell.model_key, cell.task, cell.layer),
    )
    if payload.get("status") != "ok" or payload.get("cell_slug") != cell.slug:
        raise RuntimeError(f"completed construct cell shard is invalid: {cell.slug}")
    for field in (
        "compatibility_rows",
        "compatibility_csv",
        "cell_summary",
        "group_rows",
        "group_rows_csv",
    ):
        artifact = payload.get(field)
        if not isinstance(artifact, Mapping):
            raise TypeError(f"construct cell shard lacks {field}: {cell.slug}")
        path = _run_path(context, artifact.get("path"))
        if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
            raise RuntimeError(f"construct cell artifact hash mismatch: {cell.slug}/{field}")


def run_construct_panel(
    context: RunContext,
    extension: ExtensionConfig,
    base: RevisionConfig,
    *,
    device: torch.device,
) -> dict[str, Any]:
    preflight = _read_json(context.run_dir / "preflight_report.json")
    _verify_preflight_report(context, extension, base, preflight)
    expected_keys = tuple(
        (cell.model_key, cell.task, cell.layer)
        for cell in extension.confirmatory_cells
    )
    completed = context.completed_keys(
        _CONSTRUCT_CELL_EXPERIMENT,
        expected_keys=expected_keys,
    )
    failures: list[tuple[ConstructCell, BaseException]] = []
    for cell in extension.confirmatory_cells:
        key = (cell.model_key, cell.task, cell.layer)
        if key in completed:
            _validate_completed_cell_shard(context, cell)
            continue
        try:
            summary = run_construct_cell_worker(
                context,
                base,
                cell=cell,
                device=device,
            )
            if summary.get("status") != "ok":
                raise RuntimeError(
                    f"construct worker returned status {summary.get('status')!r}"
                )
            cell_root = context.run_dir / "construct" / "cells" / cell.slug
            compatibility_path = cell_root / "construct_check_rows.parquet"
            compatibility_csv = cell_root / "construct_check_rows.csv"
            cell_summary = cell_root / "construct_check_summary.json"
            group_path, group_csv = _cell_group_paths(context.run_dir, cell)
            compatibility = pd.read_parquet(compatibility_path)
            if len(compatibility) != 390:
                raise RuntimeError(
                    f"construct cell {cell.slug} requires exactly 390 compatibility rows"
                )
            _load_generic_group_rows(context.run_dir, cell)
            context.write_json_shard(
                _CONSTRUCT_CELL_EXPERIMENT,
                key,
                {
                    "status": "ok",
                    "cell_slug": cell.slug,
                    "compatibility_rows": {
                        "path": _relative(compatibility_path, context.run_dir),
                        "sha256": sha256_file(compatibility_path),
                        "rows": len(compatibility),
                    },
                    "compatibility_csv": {
                        "path": _relative(compatibility_csv, context.run_dir),
                        "sha256": sha256_file(compatibility_csv),
                    },
                    "cell_summary": {
                        "path": _relative(cell_summary, context.run_dir),
                        "sha256": sha256_file(cell_summary),
                    },
                    "group_rows": {
                        "path": _relative(group_path, context.run_dir),
                        "sha256": sha256_file(group_path),
                    },
                    "group_rows_csv": {
                        "path": _relative(group_csv, context.run_dir),
                        "sha256": sha256_file(group_csv),
                    },
                },
            )
        except Exception as error:  # noqa: BLE001 - persist an explicit cell failure
            failure_frame = _cell_status_frame(cell, error)
            failure_root = context.run_dir / "construct" / "cells" / cell.slug
            atomic_write_parquet(failure_root / "construct_group_rows.failed.parquet", failure_frame)
            atomic_write_csv(failure_root / "construct_group_rows.failed.csv", failure_frame)
            context.write_json_shard(
                _CONSTRUCT_CELL_EXPERIMENT,
                key,
                {
                    "status": "failed",
                    "cell_slug": cell.slug,
                    "failure_stage": "construct_cell",
                    "failure_reason": f"{type(error).__name__}: {error}",
                },
            )
            failures.append((cell, error))
            break
    if failures:
        cell, error = failures[0]
        raise RuntimeError(
            f"construct panel stopped after explicit failure record for {cell.slug}: {error}"
        ) from error

    observed = context.completed_keys(
        _CONSTRUCT_CELL_EXPERIMENT,
        expected_keys=expected_keys,
    )
    assert_exact_keys(expected_keys, observed)
    for cell in extension.confirmatory_cells:
        _validate_completed_cell_shard(context, cell)

    frames = [
        _load_generic_group_rows(context.run_dir, cell)
        for cell in extension.confirmatory_cells
    ]
    pilot = build_pilot_group_rows(_base_run_path(extension), base, extension.pilot)
    pilot_parquet, pilot_csv = _cell_group_paths(context.run_dir, extension.pilot)
    atomic_write_parquet(pilot_parquet, pilot)
    atomic_write_csv(pilot_csv, pilot)
    frames.insert(0, pilot)
    consolidated = pd.concat(frames, ignore_index=True, sort=False)
    validation = validate_construct_group_rows_without_inference(
        consolidated,
        extension,
    )
    parquet_path = context.run_dir / _GROUP_ROWS_PARQUET
    csv_path = context.run_dir / _GROUP_ROWS_CSV
    atomic_write_parquet(parquet_path, consolidated)
    atomic_write_csv(csv_path, consolidated)
    report = {
        "schema_version": 1,
        "status": "ok",
        "extension_config_hash": extension.config_hash,
        "base_config_hash": base.config_hash,
        "generating_git_commit": current_git_commit(),
        "confirmatory_cells": [cell.slug for cell in extension.confirmatory_cells],
        "pilot_cell": extension.pilot.slug,
        "worker_edits": len(extension.confirmatory_cells)
        * len(base.construct_edit_keys()),
        "inferential_candidate_edits": len(extension.confirmatory_cells)
        * len([key for key in base.construct_edit_keys() if key.edit_kind == "dcand_crossfit"]),
        "compatibility_rows": 4290,
        "group_rows": {
            "path": parquet_path.name,
            "sha256": sha256_file(parquet_path),
            "csv_path": csv_path.name,
            "csv_sha256": sha256_file(csv_path),
            "rows": len(consolidated),
        },
        "pilot_group_rows": {
            "path": _relative(pilot_parquet, context.run_dir),
            "sha256": sha256_file(pilot_parquet),
            "csv_path": _relative(pilot_csv, context.run_dir),
            "csv_sha256": sha256_file(pilot_csv),
            "rows": len(pilot),
        },
        "validation": validation,
        "confirmatory_endpoints_computed": False,
    }
    atomic_write_json(context.run_dir / "construct_panel_report.json", report)
    context.update_manifest(
        {
            "construct_panel": {
                "status": "ok",
                "confirmatory_cells": len(extension.confirmatory_cells),
            }
        }
    )
    return report


def run_analyze(
    context: RunContext,
    extension: ExtensionConfig,
    base: RevisionConfig,
) -> dict[str, Any]:
    construct_report = _read_json(context.run_dir / "construct_panel_report.json")
    if construct_report.get("status") != "ok":
        raise RuntimeError("analysis requires a complete construct-panel report")
    group_record = construct_report.get("group_rows", {})
    rows_path = _run_path(context, group_record.get("path"))
    if not rows_path.is_file() or sha256_file(rows_path) != group_record.get("sha256"):
        raise RuntimeError("construct-panel normalized group rows changed before analysis")
    validation = validate_construct_group_rows_without_inference(rows_path, extension)
    construct_summary = analyze_construct_panel(rows_path, config=extension)
    robustness_path = context.run_dir / "floor_robustness_summary.json"
    robustness = _read_json(robustness_path)
    if robustness.get("status") != "ok":
        raise RuntimeError("analysis requires a successful robustness summary")
    result = {
        "schema_version": 1,
        "status": "complete",
        "extension_config_hash": extension.config_hash,
        "base_config_hash": base.config_hash,
        "generating_git_commit": current_git_commit(),
        "floor_robustness": robustness,
        "construct_panel": construct_summary,
        "validation": {
            "complete": True,
            "construct_rows_validated_before_inference": True,
            "construct_row_validation": validation,
            "pilot_descriptive_only": True,
            "confirmatory_family_size": 11,
        },
        "artifact_hashes": {
            robustness_path.name: sha256_file(robustness_path),
            rows_path.name: sha256_file(rows_path),
        },
    }
    atomic_write_json(context.run_dir / _ANALYSIS_SUMMARY, result)
    context.update_manifest({"analysis": {"status": "complete"}})
    return result


def _nonempty_run_files(context: RunContext) -> list[Path]:
    excluded = {
        context.lock_path.resolve(),
        context.manifest_path.resolve(),
        context.console_log_path.resolve(),
        (context.run_dir / "artifact_manifest.json").resolve(),
        (context.run_dir / "portability_report.json").resolve(),
    }
    return sorted(
        path
        for path in context.run_dir.rglob("*")
        if path.is_file()
        and path.stat().st_size > 0
        and path.resolve() not in excluded
        and ".failed." not in path.name
        and "failed_shards" not in path.parts
    )


def run_package_artifacts(
    context: RunContext,
    extension: ExtensionConfig,
    base: RevisionConfig,
    *,
    device: torch.device,
) -> dict[str, Any]:
    from .portability import (
        build_artifact_manifest,
        build_environment_lock,
        verify_artifact_manifest,
    )

    analysis_path = context.run_dir / _ANALYSIS_SUMMARY
    analysis = _read_json(analysis_path)
    if analysis.get("status") != "complete":
        raise RuntimeError("artifact packaging requires complete extension analysis")
    deterministic = bool(base.raw["reproducibility"]["deterministic_algorithms"])
    environment = build_environment_lock(
        spec_hash=extension.config_hash,
        git_commit=current_git_commit(),
        producing_run=context.run_id,
        device_protocol={
            "transformer_extraction": str(device),
            "fresh_decoders": "cpu",
            "statistics": "cpu",
            "cublas_workspace": (
                os.environ.get("CUBLAS_WORKSPACE_CONFIG", "missing")
                if device.type == "cuda"
                else "not_applicable"
            ),
        },
        deterministic_algorithms=deterministic,
        requirements_path=PROJECT_ROOT / "requirements.txt",
    )
    environment_path = context.run_dir / "extension_environment_lock.json"
    requirements_path = context.run_dir / "extension_environment_requirements.txt"
    atomic_write_json(environment_path, environment)
    requirement_text = "\n".join(environment["requirements"]) + "\n"
    atomic_write_via(
        requirements_path,
        lambda temporary: temporary.write_text(
            requirement_text,
            encoding="utf-8",
            newline="\n",
        ),
    )

    files = _nonempty_run_files(context)
    analysis_names = {
        "preflight_report.json",
        "floor_robustness_summary.json",
        "construct_panel_report.json",
        _GROUP_ROWS_PARQUET,
        _GROUP_ROWS_CSV,
        _ANALYSIS_SUMMARY,
        environment_path.name,
        requirements_path.name,
    }
    analysis_files = [path for path in files if _relative(path, context.run_dir) in analysis_names]
    if len(analysis_files) != len(analysis_names):
        missing = sorted(analysis_names - {_relative(path, context.run_dir) for path in analysis_files})
        raise RuntimeError(f"analysis artifact tier is incomplete: {missing}")
    audit_markers = (
        ".per-example.pt",
        "/edits/",
        "/directions/",
        "/checkpoints/",
        "split_manifest.json",
    )
    audit_extra = [
        path
        for path in files
        if any(marker in f"/{_relative(path, context.run_dir)}" for marker in audit_markers)
    ]
    audit_files = sorted(set(analysis_files) | set(audit_extra))
    bitwise_files = files
    manifest = build_artifact_manifest(
        context.run_dir,
        {
            "analysis": analysis_files,
            "audit": audit_files,
            "bitwise": bitwise_files,
        },
        producing_run=context.run_id,
    )
    manifest_path = context.run_dir / "artifact_manifest.json"
    atomic_write_json(manifest_path, manifest)
    verify_artifact_manifest(context.run_dir, manifest)
    report = {
        "schema_version": 1,
        "status": "ok",
        "extension_config_hash": extension.config_hash,
        "base_config_hash": base.config_hash,
        "environment_lock": {
            "path": environment_path.name,
            "sha256": sha256_file(environment_path),
            "aggregate_sha256": environment["aggregate_sha256"],
            "requirements_path": requirements_path.name,
            "requirements_sha256": sha256_file(requirements_path),
        },
        "artifact_manifest": {
            "path": manifest_path.name,
            "sha256": sha256_file(manifest_path),
            "aggregate_sha256": manifest["aggregate_sha256"],
            "files": len(manifest["files"]),
            "scope": (
                "Immutable scientific payload present at package-artifacts; "
                "the still-mutating run manifest and console log are excluded."
            ),
        },
    }
    atomic_write_json(context.run_dir / "portability_report.json", report)
    context.update_manifest({"package_artifacts": {"status": "ok"}})
    return report


def _figure_record(artifact: Any, context: RunContext) -> dict[str, Any]:
    if isinstance(artifact, Mapping):
        pdf_value = artifact.get("pdf_path") or artifact.get("pdf")
        png_value = artifact.get("png_path") or artifact.get("png")
        metadata = artifact.get("metadata", {})
    else:
        pdf_value = getattr(artifact, "pdf_path", None)
        png_value = getattr(artifact, "png_path", None)
        metadata = getattr(artifact, "metadata", {})
    pdf_path = Path(pdf_value) if pdf_value is not None else None
    png_path = Path(png_value) if png_value is not None else None
    if pdf_path is None or png_path is None:
        raise RuntimeError("extension figure API did not return PDF and PNG paths")
    for path in (pdf_path, png_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"extension figure is missing or empty: {path}")
    project_directory = PROJECT_ROOT / "figures"
    project_directory.mkdir(parents=True, exist_ok=True)
    project_pdf = _atomic_copy(pdf_path, project_directory / pdf_path.name)
    project_png = _atomic_copy(png_path, project_directory / png_path.name)
    return {
        "run_pdf": _relative(pdf_path, context.run_dir),
        "run_png": _relative(png_path, context.run_dir),
        "pdf_sha256": sha256_file(pdf_path),
        "png_sha256": sha256_file(png_path),
        "project_pdf": _relative(project_pdf, PROJECT_ROOT),
        "project_png": _relative(project_png, PROJECT_ROOT),
        "project_pdf_sha256": sha256_file(project_pdf),
        "project_png_sha256": sha256_file(project_png),
        "metadata": sanitize_manifest_payload(metadata),
    }


def run_figures(
    context: RunContext,
    extension: ExtensionConfig,
    base: RevisionConfig,
) -> dict[str, Any]:
    del extension, base
    summary_path = context.run_dir / _ANALYSIS_SUMMARY
    if _read_json(summary_path).get("status") != "complete":
        raise RuntimeError("figures require complete extension analysis")
    output_directory = context.run_dir / "figures"
    floor = create_floor_sensitivity_figure(
        summary_path,
        output_directory,
        stem="fig_floor_sensitivity",
    )
    construct = create_construct_panel_figure(
        summary_path,
        output_directory,
        stem="fig_construct_panel",
    )
    report = {
        "schema_version": 1,
        "status": "ok",
        "analysis_sha256": sha256_file(summary_path),
        "figures": {
            "floor_sensitivity": _figure_record(floor, context),
            "construct_panel": _figure_record(construct, context),
        },
    }
    atomic_write_json(context.run_dir / "extension_figures_report.json", report)
    context.update_manifest({"figures": {"status": "ok", "count": 2}})
    return report


def _paper_patch_payload(report: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(report) and not isinstance(report, type):
        return sanitize_manifest_payload(dataclasses.asdict(report))
    if isinstance(report, Mapping):
        return sanitize_manifest_payload(dict(report))
    return {"report": str(report)}


def validate_workshop_submission(
    context: RunContext,
    compilation: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the LP4FM source, checklist, and compiled PDF gates."""

    source_path = PROJECT_ROOT / "main_revised.tex"
    checklist_path = PROJECT_ROOT / "neurips_2026_checklist.tex"
    source = source_path.read_text(encoding="utf-8")
    checklist = checklist_path.read_text(encoding="utf-8")
    required_source = (
        r"\usepackage[dblblindworkshop]{neurips_2026}",
        r"\author{Anonymous Authors}",
        r"\workshoptitle{Linguistic Principles for Foundation Models}",
        r"\section{Scope, responsible use, and conclusion}",
        r"\bibliography{references}",
        r"\appendix",
        r"\input{neurips_2026_checklist.tex}",
    )
    missing = [marker for marker in required_source if marker not in source]
    if missing:
        raise RuntimeError(f"workshop source is missing required markers: {missing}")
    references_at = source.index(r"\bibliography{references}")
    appendix_at = source.index(r"\appendix")
    checklist_at = source.index(r"\input{neurips_2026_checklist.tex}")
    if not references_at < appendix_at < checklist_at:
        raise RuntimeError("workshop source must order references, appendices, then checklist")
    if "TODO" in source.upper() or "TODO" in checklist.upper():
        raise RuntimeError("workshop source or checklist contains a TODO")
    if r"\section*{NeurIPS Paper Checklist}" not in checklist:
        raise RuntimeError("workshop checklist heading is missing")
    answer_count = sum(
        1
        for line in checklist.splitlines()
        if line.strip().startswith(r"\item[] Answer:")
    )
    if answer_count != 16:
        raise RuntimeError(
            f"workshop checklist must contain exactly 16 answered items, got {answer_count}"
        )
    if (
        compilation.get("status") != "compiled"
        or type(compilation.get("main_text_pages")) is not int
        or compilation["main_text_pages"] > 9
        or compilation.get("visual_inspection") != "pending"
    ):
        raise RuntimeError("compiled workshop paper failed the nine-page or status gate")
    pdf_path = context.run_dir / "paper" / "main_revised.pdf"
    if not pdf_path.is_file() or pdf_path.stat().st_size <= 0:
        raise RuntimeError("compiled workshop PDF is missing or empty")
    maximum_bytes = 50 * 1024 * 1024
    if pdf_path.stat().st_size > maximum_bytes:
        raise RuntimeError("compiled workshop PDF exceeds the 50 MB submission limit")
    if sha256_file(pdf_path) != compilation.get("pdf_sha256"):
        raise RuntimeError("compiled workshop PDF hash differs from compilation record")
    extracted = subprocess.run(
        ["pdftotext", str(pdf_path), "-"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        timeout=120,
    )
    if extracted.returncode != 0:
        raise RuntimeError("could not extract compiled workshop PDF text")
    pdf_text = extracted.stdout.decode("utf-8", errors="replace")
    required_pdf_text = (
        "References",
        "NeurIPS Paper Checklist",
    )
    if any(marker not in pdf_text for marker in required_pdf_text):
        raise RuntimeError("compiled workshop PDF lacks references or checklist text")
    return {
        "style": "dblblindworkshop",
        "anonymous_authors": True,
        "responsible_use_section": True,
        "checklist_answered_items": answer_count,
        "references_appendices_checklist_order": True,
        "main_text_pages": compilation["main_text_pages"],
        "pdf_bytes": pdf_path.stat().st_size,
        "pdf_under_50_mb": True,
        "visual_inspection": "pending",
    }


def run_patch_paper(
    context: RunContext,
    extension: ExtensionConfig,
    base: RevisionConfig,
) -> dict[str, Any]:
    del extension, base
    summary_path = context.run_dir / _ANALYSIS_SUMMARY
    figures_path = context.run_dir / "extension_figures_report.json"
    if _read_json(summary_path).get("status") != "complete":
        raise RuntimeError("paper patch requires complete extension analysis")
    if _read_json(figures_path).get("status") != "ok":
        raise RuntimeError("paper patch requires successful extension figures")
    project_source = PROJECT_ROOT / "main_revised.tex"
    snapshot = context.run_dir / "paper" / "main_revised.preextension.tex"
    if not snapshot.is_file():
        _atomic_copy(project_source, snapshot)
    project_macros = PROJECT_ROOT / "extension_numbers.tex"
    patch_report = patch_manuscript_with_extension(
        snapshot,
        project_source,
        summary_path,
        macros_path=project_macros,
    )
    if not project_source.is_file() or project_source.stat().st_size <= 0:
        raise RuntimeError("paper patch did not produce main_revised.tex")
    if not project_macros.is_file() or project_macros.stat().st_size <= 0:
        raise RuntimeError("paper patch did not produce extension numeric macros")
    run_macros = _atomic_copy(
        project_macros,
        context.run_dir / "paper" / project_macros.name,
    )
    run_source = _atomic_copy(
        project_source,
        context.run_dir / "paper" / "main_revised.tex",
    )
    compilation = compile_workshop_manuscript(context)
    workshop_validation = validate_workshop_submission(context, compilation)
    report = {
        "schema_version": 1,
        "status": "compiled_pending_visual_inspection",
        "analysis_sha256": sha256_file(summary_path),
        "figures_report_sha256": sha256_file(figures_path),
        "source_snapshot": _relative(snapshot, context.run_dir),
        "source_snapshot_sha256": sha256_file(snapshot),
        "project_source": _relative(project_source, PROJECT_ROOT),
        "project_source_sha256": sha256_file(project_source),
        "run_source": _relative(run_source, context.run_dir),
        "run_source_sha256": sha256_file(run_source),
        "macros_path": _relative(run_macros, context.run_dir),
        "macros_sha256": sha256_file(run_macros),
        "patch": _paper_patch_payload(patch_report),
        "compilation": compilation,
        "workshop_validation": workshop_validation,
    }
    atomic_write_json(context.run_dir / "paper_extension_report.json", report)
    context.update_manifest(
        {
            "paper_extension": {
                "status": "compiled_pending_visual_inspection",
                "main_text_pages": compilation["main_text_pages"],
                "pdf_sha256": compilation["pdf_sha256"],
            }
        }
    )
    return report


def _verify_hashed_record(context: RunContext, record: Mapping[str, Any]) -> Path:
    path = _run_path(context, record.get("path"))
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise RuntimeError(f"artifact hash mismatch: {record.get('path')}")
    return path


def stage_is_complete(
    context: RunContext,
    stage: str,
    *,
    extension: ExtensionConfig,
    base: RevisionConfig,
) -> bool:
    """Validate a completed stage from bytes; never trust manifest status alone."""

    if stage not in _STAGE_REPORTS:
        raise ValueError(f"unknown extension stage: {stage!r}")
    filename, accepted = _STAGE_REPORTS[stage]
    path = context.run_dir / filename
    if not path.is_file():
        return False
    report = _read_json(path)
    if report.get("status") not in accepted:
        return False
    if stage == "preflight":
        _verify_preflight_report(context, extension, base, report)
    elif stage == "robustness":
        source = _base_run_path(extension) / "matched_split_rows.parquet"
        if report.get("source_rows_sha256") != sha256_file(source):
            raise RuntimeError("completed robustness source hash changed")
    elif stage == "construct-panel":
        if report.get("confirmatory_endpoints_computed") is not False:
            raise RuntimeError("construct-panel report computed endpoints before analysis")
        group_record = report.get("group_rows", {})
        _verify_hashed_record(context, group_record)
        group_csv = _run_path(context, group_record.get("csv_path"))
        if not group_csv.is_file() or sha256_file(group_csv) != group_record.get(
            "csv_sha256"
        ):
            raise RuntimeError("construct-panel normalized CSV hash mismatch")
        pilot_record = report.get("pilot_group_rows", {})
        _verify_hashed_record(context, pilot_record)
        pilot_csv = _run_path(context, pilot_record.get("csv_path"))
        if not pilot_csv.is_file() or sha256_file(pilot_csv) != pilot_record.get(
            "csv_sha256"
        ):
            raise RuntimeError("construct-panel pilot CSV hash mismatch")
        expected = [
            (cell.model_key, cell.task, cell.layer)
            for cell in extension.confirmatory_cells
        ]
        observed = context.completed_keys(
            _CONSTRUCT_CELL_EXPERIMENT,
            expected_keys=expected,
        )
        assert_exact_keys(expected, observed)
        for cell in extension.confirmatory_cells:
            _validate_completed_cell_shard(context, cell)
    elif stage == "analyze":
        if report.get("validation", {}).get("complete") is not True:
            return False
        for relative, digest in report.get("artifact_hashes", {}).items():
            artifact = _run_path(context, relative)
            if not artifact.is_file() or sha256_file(artifact) != digest:
                raise RuntimeError(f"analysis input changed: {relative}")
    elif stage == "package-artifacts":
        from .portability import verify_artifact_manifest

        environment = _verify_hashed_record(context, report.get("environment_lock", {}))
        environment_payload = _read_json(environment)
        if environment_payload.get("aggregate_sha256") != report.get(
            "environment_lock", {}
        ).get("aggregate_sha256"):
            raise RuntimeError("environment lock aggregate hash mismatch")
        manifest_path = _verify_hashed_record(context, report.get("artifact_manifest", {}))
        manifest = _read_json(manifest_path)
        verify_artifact_manifest(context.run_dir, manifest)
    elif stage == "figures":
        analysis_path = context.run_dir / _ANALYSIS_SUMMARY
        if not analysis_path.is_file() or sha256_file(analysis_path) != report.get(
            "analysis_sha256"
        ):
            raise RuntimeError("completed figures analysis hash changed")
        for artifact in report.get("figures", {}).values():
            for path_key, hash_key in (
                ("run_pdf", "pdf_sha256"),
                ("run_png", "png_sha256"),
            ):
                figure = _run_path(context, artifact.get(path_key))
                if not figure.is_file() or sha256_file(figure) != artifact.get(hash_key):
                    raise RuntimeError(f"completed figure changed: {artifact.get(path_key)}")
    elif stage == "patch-paper":
        run_source = _run_path(context, report.get("run_source"))
        if not run_source.is_file() or sha256_file(run_source) != report.get(
            "run_source_sha256"
        ):
            raise RuntimeError("completed extension paper source changed")
        macros = _run_path(context, report.get("macros_path"))
        if not macros.is_file() or sha256_file(macros) != report.get("macros_sha256"):
            raise RuntimeError("completed extension numeric macros changed")
        validation = validate_workshop_submission(
            context,
            report.get("compilation", {}),
        )
        if validation != report.get("workshop_validation"):
            raise RuntimeError("completed workshop validation report changed")
        project_source = PROJECT_ROOT / str(report.get("project_source"))
        if not project_source.is_file() or sha256_file(project_source) != report.get(
            "project_source_sha256"
        ):
            raise RuntimeError("completed project manuscript source changed")
    return True


def _validate_stage_result(stage: str, result: Mapping[str, Any]) -> None:
    accepted = _STAGE_REPORTS[stage][1]
    status = result.get("status")
    if status not in accepted:
        raise RuntimeError(
            f"{stage} returned status {status!r}; expected one of {sorted(accepted)!r}"
        )
    if stage == "analyze" and result.get("validation", {}).get("complete") is not True:
        raise RuntimeError("extension analysis did not pass its complete validation gate")


def run_stage(
    stage: str,
    context: RunContext,
    extension: ExtensionConfig,
    base: RevisionConfig,
    *,
    device: torch.device,
    allow_reproduced_base: bool = False,
) -> Mapping[str, Any]:
    if stage == "preflight":
        return run_preflight(
            context,
            extension,
            base,
            device=device,
            allow_reproduced_base=allow_reproduced_base,
        )
    if stage == "robustness":
        return run_robustness(context, extension, base)
    if stage == "construct-panel":
        return run_construct_panel(context, extension, base, device=device)
    if stage == "analyze":
        return run_analyze(context, extension, base)
    if stage == "package-artifacts":
        return run_package_artifacts(context, extension, base, device=device)
    if stage == "figures":
        return run_figures(context, extension, base)
    if stage == "patch-paper":
        return run_patch_paper(context, extension, base)
    raise ValueError(f"unknown extension stage: {stage!r}")


def configure_base_reproducibility(base: RevisionConfig) -> None:
    from .runner import _configure_reproducibility as implementation

    implementation(base)


def configure_extension_reproducibility(
    base: RevisionConfig,
    device: torch.device,
) -> None:
    """Apply the base protocol plus deterministic CUDA workspace settings."""

    if device.type == "cuda":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    configure_base_reproducibility(base)
    if device.type == "cuda" and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _validate_resume_manifest(
    context: RunContext,
    extension: ExtensionConfig,
    base: RevisionConfig,
) -> None:
    manifest = _read_json(context.manifest_path)
    if manifest.get("extension_config") != extension.to_record():
        raise RuntimeError("resume extension config record differs from the locked spec")
    if manifest.get("base_config_hash") != base.config_hash:
        raise RuntimeError("resume base config hash differs from the locked base spec")


def _predecessors(stage: str) -> tuple[str, ...]:
    return STAGES[: STAGES.index(stage)]


def _pipeline_final_status(final_stage: str) -> str:
    if final_stage == "patch-paper":
        return "pending_visual_inspection"
    return "stage_complete"


def execute(
    command: str,
    extension: ExtensionConfig,
    base: RevisionConfig,
    args: Any,
) -> int:
    """Execute one stage or the full extension with exact, fail-closed resume."""

    if command != "all" and command not in STAGES:
        raise ValueError(f"unknown extension command: {command!r}")
    extension.validate()
    base_run_override = getattr(args, "base_run", None)
    allow_reproduced_base = bool(getattr(args, "allow_reproduced_base", False))
    if allow_reproduced_base and base_run_override is None:
        raise ValueError("--allow-reproduced-base requires --base-run")
    if base_run_override is not None:
        resolved_base = (PROJECT_ROOT / Path(base_run_override)).resolve()
        if not resolved_base.is_relative_to(PROJECT_ROOT.resolve()):
            raise ValueError("--base-run must resolve inside the repository")
        extension = dataclasses.replace(
            extension,
            base_run=resolved_base.relative_to(PROJECT_ROOT.resolve()),
        )
    device = select_extension_device(str(args.device))
    configure_extension_reproducibility(base, device)
    commit = current_git_commit()
    output_root = Path(args.output_root)
    if bool(args.resume):
        run_dir = resolve_extension_resume_directory(
            output_root,
            config_hash=extension.config_hash,
            git_commit=commit,
        )
        context = RunContext.resume(
            run_dir,
            config_hash=extension.config_hash,
            git_commit=commit,
        )
        _validate_resume_manifest(context, extension, base)
    else:
        context = RunContext.create(
            output_root=output_root,
            config_hash=extension.config_hash,
            git_commit=commit,
            manifest={
                "extension_config": extension.to_record(),
                "base_config_hash": base.config_hash,
                "extension_config_source": Path(args.config).name,
                "base_config_source": Path(args.base_config).name,
                "requested_device": str(args.device),
                "selected_device": str(device),
                "confirmatory_endpoints_computed_at_creation": False,
            },
        )

    file_handler = logging.FileHandler(context.console_log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    stages = STAGES if command == "all" else (command,)
    completed: list[str] = []
    skipped: list[str] = []
    started = _utc_now()
    try:
        with context:
            context.update_manifest(
                {"pipeline": {"status": "running", "command": command, "started_at_utc": started}}
            )
            for stage in stages:
                for predecessor in _predecessors(stage):
                    if predecessor in completed or predecessor in skipped:
                        continue
                    if not stage_is_complete(
                        context,
                        predecessor,
                        extension=extension,
                        base=base,
                    ):
                        raise RuntimeError(
                            f"{stage} requires completed predecessor {predecessor}"
                        )
                if bool(args.resume) and stage_is_complete(
                    context,
                    stage,
                    extension=extension,
                    base=base,
                ):
                    skipped.append(stage)
                    continue
                LOGGER.info("%s: starting", stage)
                result = run_stage(
                    stage,
                    context,
                    extension,
                    base,
                    device=device,
                    allow_reproduced_base=allow_reproduced_base,
                )
                _validate_stage_result(stage, result)
                completed.append(stage)
                context.update_manifest(
                    {
                        "pipeline": {
                            "status": "running",
                            "command": command,
                            "started_at_utc": started,
                            "last_completed_stage": stage,
                            "completed_this_invocation": completed,
                            "skipped_validated": skipped,
                        }
                    }
                )
                LOGGER.info("%s: completed", stage)
            context.update_manifest(
                {
                    "pipeline": {
                        "status": _pipeline_final_status(stages[-1]),
                        "command": command,
                        "started_at_utc": started,
                        "finished_at_utc": _utc_now(),
                        "last_completed_stage": stages[-1],
                        "completed_this_invocation": completed,
                        "skipped_validated": skipped,
                    }
                }
            )
        return 0
    except BaseException as exc:
        try:
            manifest = _read_json(context.manifest_path)
            manifest["pipeline"] = sanitize_manifest_payload(
                {
                    "status": "failed",
                    "command": command,
                    "started_at_utc": started,
                    "failed_at_utc": _utc_now(),
                    "last_completed_stage": completed[-1] if completed else None,
                    "completed_this_invocation": completed,
                    "skipped_validated": skipped,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            atomic_write_json(context.manifest_path, manifest)
        finally:
            context.close()
        raise
    finally:
        root_logger.removeHandler(file_handler)
        file_handler.close()


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "STAGES",
    "analyze_construct_panel",
    "base_provenance_mode",
    "build_dry_run_plan",
    "build_pilot_group_rows",
    "configure_extension_reproducibility",
    "current_git_commit",
    "execute",
    "planned_alterrep_keys",
    "projected_construct_storage_bytes",
    "require_construct_disk_capacity",
    "resolve_extension_resume_directory",
    "run_analyze",
    "run_construct_panel",
    "run_package_artifacts",
    "run_preflight",
    "run_robustness",
    "select_extension_device",
    "stage_is_complete",
    "validate_construct_cache_pins",
    "validate_construct_group_rows_without_inference",
    "validate_workshop_submission",
]
