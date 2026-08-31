"""Validation for the prospective reviewer-caveat extension specification."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import yaml

from .artifacts import sha256_json


class ExtensionConfigError(ValueError):
    """Raised when the extension specification differs from its locked design."""


@dataclass(frozen=True, order=True)
class ConstructCell:
    model_key: str
    model_id: str
    task: str
    layer: int

    @property
    def slug(self) -> str:
        return f"{self.model_key}-{self.task}-l{self.layer}"


_LOCKED_PILOT = ConstructCell("qwen", "Qwen/Qwen2.5-1.5B", "sst2", 14)
_LOCKED_CONFIRMATORY_CELLS = (
    ConstructCell("pythia", "EleutherAI/pythia-160m", "sva", 6),
    ConstructCell("pythia", "EleutherAI/pythia-160m", "sst2", 6),
    ConstructCell("gpt2", "openai-community/gpt2", "sva", 6),
    ConstructCell("gpt2", "openai-community/gpt2", "sst2", 6),
    ConstructCell("bert", "google-bert/bert-base-uncased", "sva", 6),
    ConstructCell("bert", "google-bert/bert-base-uncased", "sst2", 6),
    ConstructCell("qwen", "Qwen/Qwen2.5-1.5B", "sva", 14),
    ConstructCell("gemma", "google/gemma-2-2b", "sva", 14),
    ConstructCell("gemma", "google/gemma-2-2b", "sst2", 14),
    ConstructCell("llama", "meta-llama/Llama-3.2-3B", "sva", 14),
    ConstructCell("llama", "meta-llama/Llama-3.2-3B", "sst2", 14),
)
_LOCKED_THRESHOLDS = {
    "accuracy": 0.55,
    "target_recovery_ratio": 0.50,
    "control_retention_ratio": 0.80,
}
_LOCKED_BASE_RUN = (
    "results/reviewer_revision_2026_08/20260830T024520Z-7aab3eb"
)
_LOCKED_OUTPUT_ROOT = "results/reviewer_caveat_extension_2026_08"
_LOCKED_BOOTSTRAP_DRAWS = 10_000
_LOCKED_BOOTSTRAP_SEED = 20260830
_LOCKED_DISK_RESERVE_GIB = 8.0
_SLUG_TOKEN = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class ExtensionConfig:
    source_path: Path
    config_hash: str
    base_run: Path
    pilot: ConstructCell
    confirmatory_cells: tuple[ConstructCell, ...]
    recovery_thresholds: Mapping[str, float]
    bootstrap_draws: int
    bootstrap_seed: int
    disk_reserve_gib: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "confirmatory_cells", tuple(self.confirmatory_cells))
        object.__setattr__(
            self,
            "recovery_thresholds",
            MappingProxyType(dict(self.recovery_thresholds)),
        )

    @property
    def all_cells(self) -> tuple[ConstructCell, ...]:
        return (self.pilot, *self.confirmatory_cells)

    def to_record(self) -> dict[str, Any]:
        """Return a detached, portable JSON record for manifests and resumes."""

        def cell_record(cell: ConstructCell) -> dict[str, str | int]:
            return {
                "model_key": cell.model_key,
                "model_id": cell.model_id,
                "task": cell.task,
                "layer": cell.layer,
            }

        return {
            "config_hash": self.config_hash,
            "base_run": self.base_run.as_posix(),
            "pilot": cell_record(self.pilot),
            "confirmatory_cells": [
                cell_record(cell) for cell in self.confirmatory_cells
            ],
            "recovery_thresholds": dict(self.recovery_thresholds),
            "bootstrap_draws": self.bootstrap_draws,
            "bootstrap_seed": self.bootstrap_seed,
            "disk_reserve_gib": self.disk_reserve_gib,
        }

    def validate(self) -> None:
        if not isinstance(self.source_path, Path) or not self.source_path.is_absolute():
            raise ExtensionConfigError("source_path must be an absolute Path")
        if not isinstance(self.config_hash, str) or not _SHA256.fullmatch(
            self.config_hash
        ):
            raise ExtensionConfigError("config_hash must be a lowercase SHA-256 digest")
        if not isinstance(self.base_run, Path):
            raise ExtensionConfigError("base_run must be a Path")
        if self.base_run != Path(_LOCKED_BASE_RUN):
            raise ExtensionConfigError("base_run differs from the locked immutable run")
        if not isinstance(self.pilot, ConstructCell):
            raise ExtensionConfigError("pilot must be a ConstructCell")
        if any(
            not isinstance(cell, ConstructCell) for cell in self.confirmatory_cells
        ):
            raise ExtensionConfigError(
                "confirmatory cells must be ConstructCell values"
            )

        for location, cell in (
            ("construct_panel.pilot", self.pilot),
            *(
                (f"construct_panel.confirmatory_cells[{index}]", cell)
                for index, cell in enumerate(self.confirmatory_cells)
            ),
        ):
            _validate_cell_value(cell, location)

        pilot_identity = (
            self.pilot.model_key,
            self.pilot.task,
            self.pilot.layer,
        )
        if any(
            (cell.model_key, cell.task, cell.layer) == pilot_identity
            for cell in self.confirmatory_cells
        ):
            raise ExtensionConfigError(
                "pilot cell must not appear in confirmatory cells"
            )
        if (
            len(self.confirmatory_cells) != 11
            or len(set(self.confirmatory_cells)) != 11
        ):
            raise ExtensionConfigError(
                "exactly 11 unique confirmatory cells are required"
            )
        slugs = [cell.slug for cell in self.all_cells]
        if len(slugs) != len(set(slugs)):
            raise ExtensionConfigError("duplicate construct cell slug")
        if self.pilot != _LOCKED_PILOT:
            raise ExtensionConfigError("pilot differs from the locked Qwen/SST-2 cell")
        if self.confirmatory_cells != _LOCKED_CONFIRMATORY_CELLS:
            raise ExtensionConfigError(
                "cells differ from the locked 12-cell middle-layer panel"
            )

        thresholds = dict(self.recovery_thresholds)
        if set(thresholds) != set(_LOCKED_THRESHOLDS) or any(
            type(value) is not float or not math.isfinite(value)
            for value in thresholds.values()
        ):
            raise ExtensionConfigError(
                "recovery thresholds differ from the locked design"
            )
        if thresholds != _LOCKED_THRESHOLDS:
            raise ExtensionConfigError(
                "recovery thresholds differ from the locked design"
            )
        if type(self.bootstrap_draws) is not int:
            raise ExtensionConfigError("bootstrap draws must be an integer")
        if self.bootstrap_draws != _LOCKED_BOOTSTRAP_DRAWS:
            raise ExtensionConfigError("bootstrap requires exactly 10,000 draws")
        if type(self.bootstrap_seed) is not int:
            raise ExtensionConfigError("bootstrap seed must be an integer")
        if self.bootstrap_seed != _LOCKED_BOOTSTRAP_SEED:
            raise ExtensionConfigError("bootstrap seed differs from the locked design")
        if type(self.disk_reserve_gib) is not float or not math.isfinite(
            self.disk_reserve_gib
        ):
            raise ExtensionConfigError("disk reserve must be a finite float")
        if self.disk_reserve_gib != _LOCKED_DISK_RESERVE_GIB:
            raise ExtensionConfigError("storage requires an 8 GiB disk reserve")


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ExtensionConfigError(
                f"unhashable YAML mapping key: {key!r}"
            ) from exc
        if duplicate:
            raise ExtensionConfigError(f"duplicate YAML mapping key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _require_exact_keys(
    value: Mapping[Any, Any], expected: set[str], location: str
) -> None:
    actual = set(value)
    missing = expected - actual
    unexpected = actual - expected
    if missing:
        names = ", ".join(sorted(missing))
        raise ExtensionConfigError(
            f"{location} is missing required fields: {names}"
        )
    if unexpected:
        names = ", ".join(sorted(repr(key) for key in unexpected))
        raise ExtensionConfigError(f"{location} has unexpected fields: {names}")


def _mapping(value: Any, location: str) -> dict[Any, Any]:
    if not isinstance(value, dict):
        raise ExtensionConfigError(f"{location} must be a mapping")
    return value


def _sequence(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ExtensionConfigError(f"{location} must be a sequence")
    return value


def _locked_scalar(
    value: Any, expected: Any, location: str, *, label: str | None = None
) -> None:
    if type(value) is not type(expected) or value != expected:
        description = label if label is not None else repr(expected)
        raise ExtensionConfigError(f"{location} must be locked to {description}")


def _portable_relative_path(value: Any, location: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise ExtensionConfigError(f"{location} must be a portable relative path")
    if "\\" in value or re.match(r"^[A-Za-z]:", value):
        raise ExtensionConfigError(f"{location} must be a portable relative path")
    pure = PurePosixPath(value)
    raw_parts = value.split("/")
    if (
        pure.is_absolute()
        or any(part in {"", ".", ".."} for part in raw_parts)
        or pure.as_posix() != value
    ):
        raise ExtensionConfigError(f"{location} must be a portable relative path")
    return value, Path(*pure.parts)


def _cell_from_raw(value: Any, location: str) -> ConstructCell:
    raw = _mapping(value, location)
    _require_exact_keys(raw, {"model_key", "model_id", "task", "layer"}, location)
    for field in ("model_key", "model_id", "task"):
        if not isinstance(raw[field], str) or not raw[field]:
            raise ExtensionConfigError(f"{location}.{field} must be a non-empty string")
    if type(raw["layer"]) is not int:
        raise ExtensionConfigError(f"{location}.layer must be an integer")
    cell = ConstructCell(
        model_key=raw["model_key"],
        model_id=raw["model_id"],
        task=raw["task"],
        layer=raw["layer"],
    )
    _validate_cell_value(cell, location)
    return cell


def _validate_cell_value(cell: ConstructCell, location: str) -> None:
    if not isinstance(cell.model_key, str) or not _SLUG_TOKEN.fullmatch(
        cell.model_key
    ):
        raise ExtensionConfigError(f"{location}.model_key must be path-safe")
    if not isinstance(cell.task, str) or not _SLUG_TOKEN.fullmatch(cell.task):
        raise ExtensionConfigError(f"{location}.task must be path-safe")
    if not isinstance(cell.model_id, str) or not _MODEL_ID.fullmatch(cell.model_id):
        raise ExtensionConfigError(f"{location}.model_id must be a valid model ID")
    if (
        cell.model_id.startswith("/")
        or "//" in cell.model_id
        or ".." in cell.model_id.split("/")
    ):
        raise ExtensionConfigError(f"{location}.model_id must be a valid model ID")
    if type(cell.layer) is not int or cell.layer <= 0:
        raise ExtensionConfigError(f"{location}.layer must be a positive integer")
    if not _SLUG_TOKEN.fullmatch(cell.slug):
        raise ExtensionConfigError(f"{location} produces a path-unsafe slug")


def _validated_payload(raw: dict[Any, Any]) -> dict[str, Any]:
    root_keys = {
        "schema_version",
        "run_name",
        "purpose",
        "base_run",
        "construct_panel",
        "inference",
        "storage",
        "outputs",
    }
    _require_exact_keys(raw, root_keys, "root")
    _locked_scalar(raw["schema_version"], 1, "schema_version")
    _locked_scalar(
        raw["run_name"],
        "reviewer_caveat_extension_2026_08",
        "run_name",
    )
    if not isinstance(raw["purpose"], str) or not raw["purpose"].strip():
        raise ExtensionConfigError("purpose must be a non-empty string")

    base_run_text, base_run = _portable_relative_path(raw["base_run"], "base_run")
    if base_run_text != _LOCKED_BASE_RUN:
        raise ExtensionConfigError("base_run differs from the locked immutable run")

    panel = _mapping(raw["construct_panel"], "construct_panel")
    _require_exact_keys(
        panel,
        {"pilot", "confirmatory_cells", "recovery_thresholds"},
        "construct_panel",
    )
    pilot = _cell_from_raw(panel["pilot"], "construct_panel.pilot")
    cells_raw = _sequence(
        panel["confirmatory_cells"], "construct_panel.confirmatory_cells"
    )
    confirmatory_cells = tuple(
        _cell_from_raw(value, f"construct_panel.confirmatory_cells[{index}]")
        for index, value in enumerate(cells_raw)
    )

    thresholds_raw = _mapping(
        panel["recovery_thresholds"], "construct_panel.recovery_thresholds"
    )
    _require_exact_keys(
        thresholds_raw,
        set(_LOCKED_THRESHOLDS),
        "construct_panel.recovery_thresholds",
    )
    for name, expected in _LOCKED_THRESHOLDS.items():
        _locked_scalar(
            thresholds_raw[name],
            expected,
            f"construct_panel.recovery_thresholds.{name}",
            label="the locked recovery thresholds",
        )

    inference = _mapping(raw["inference"], "inference")
    _require_exact_keys(
        inference,
        {"bootstrap", "alternative", "multiplicity", "family_size"},
        "inference",
    )
    bootstrap = _mapping(inference["bootstrap"], "inference.bootstrap")
    _require_exact_keys(
        bootstrap,
        {
            "draws",
            "seed",
            "resampling_unit",
            "shared_resamples_across_models_within_task",
        },
        "inference.bootstrap",
    )
    if type(bootstrap["draws"]) is not int:
        raise ExtensionConfigError("inference.bootstrap.draws must be an integer")
    if bootstrap["draws"] != _LOCKED_BOOTSTRAP_DRAWS:
        raise ExtensionConfigError(
            "inference.bootstrap.draws must be exactly 10,000"
        )
    _locked_scalar(
        bootstrap["seed"],
        _LOCKED_BOOTSTRAP_SEED,
        "inference.bootstrap.seed",
    )
    _locked_scalar(
        bootstrap["resampling_unit"],
        "final_test_group_id",
        "inference.bootstrap.resampling_unit",
    )
    _locked_scalar(
        bootstrap["shared_resamples_across_models_within_task"],
        True,
        "inference.bootstrap.shared_resamples_across_models_within_task",
    )
    _locked_scalar(
        inference["alternative"],
        "one_sided",
        "inference.alternative",
        label="one_sided inference",
    )
    _locked_scalar(
        inference["multiplicity"],
        "holm",
        "inference.multiplicity",
        label="Holm adjustment",
    )
    _locked_scalar(inference["family_size"], 11, "inference.family_size")

    storage = _mapping(raw["storage"], "storage")
    _require_exact_keys(storage, {"disk_reserve_gib"}, "storage")
    reserve = storage["disk_reserve_gib"]
    if type(reserve) is not float or not math.isfinite(reserve):
        raise ExtensionConfigError("storage.disk_reserve_gib must be a finite float")
    if reserve != _LOCKED_DISK_RESERVE_GIB:
        raise ExtensionConfigError("storage requires an 8 GiB disk reserve")

    outputs = _mapping(raw["outputs"], "outputs")
    _require_exact_keys(outputs, {"root", "immutable_run_directory"}, "outputs")
    output_text, _ = _portable_relative_path(outputs["root"], "outputs.root")
    if output_text != _LOCKED_OUTPUT_ROOT:
        raise ExtensionConfigError(
            f"outputs.root must be {_LOCKED_OUTPUT_ROOT}"
        )
    _locked_scalar(
        outputs["immutable_run_directory"],
        True,
        "outputs.immutable_run_directory",
        label="true for immutable run directories",
    )

    return {
        "base_run": base_run,
        "pilot": pilot,
        "confirmatory_cells": confirmatory_cells,
        "recovery_thresholds": dict(thresholds_raw),
        "bootstrap_draws": bootstrap["draws"],
        "bootstrap_seed": bootstrap["seed"],
        "disk_reserve_gib": reserve,
    }


def load_extension_config(path: str | Path) -> ExtensionConfig:
    """Load and validate a prospectively locked extension YAML file."""

    source_path = Path(path)
    try:
        raw = yaml.load(
            source_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader
        )
    except ExtensionConfigError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ExtensionConfigError(f"could not load extension config: {exc}") from exc
    if not isinstance(raw, dict):
        raise ExtensionConfigError("extension config root must be a mapping")

    values = _validated_payload(raw)
    try:
        config_hash = sha256_json(raw)
    except (TypeError, ValueError) as exc:
        raise ExtensionConfigError(
            f"extension config is not canonically hashable: {exc}"
        ) from exc
    config = ExtensionConfig(
        source_path=source_path.resolve(),
        config_hash=config_hash,
        **values,
    )
    config.validate()
    return config


__all__ = [
    "ConstructCell",
    "ExtensionConfig",
    "ExtensionConfigError",
    "load_extension_config",
]
