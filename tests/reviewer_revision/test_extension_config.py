from __future__ import annotations

import json
import re
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.reviewer_revision.artifacts import sha256_json
from src.reviewer_revision.extension_config import (
    ConstructCell,
    ExtensionConfigError,
    load_extension_config,
)

SPEC_PATH = Path(__file__).resolve().parents[2] / "revision_caveat_extension_spec.yaml"

PILOT = ConstructCell("qwen", "Qwen/Qwen2.5-1.5B", "sst2", 14)
CONFIRMATORY_CELLS = (
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


def _valid_extension_payload() -> dict[str, Any]:
    payload = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_yaml(tmp_path: Path, payload: Any) -> Path:
    path = tmp_path / "extension.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _set_path(payload: dict[str, Any], path: tuple[str | int, ...], value: Any) -> None:
    cursor: Any = payload
    for component in path[:-1]:
        cursor = cursor[component]
    cursor[path[-1]] = value


def test_extension_config_locks_eleven_untouched_cells() -> None:
    config = load_extension_config(SPEC_PATH)

    assert config.pilot == PILOT
    assert config.confirmatory_cells == CONFIRMATORY_CELLS
    assert len(config.confirmatory_cells) == 11
    assert config.pilot not in config.confirmatory_cells
    assert config.all_cells == (PILOT, *CONFIRMATORY_CELLS)
    assert len(config.all_cells) == 12
    assert len({cell.slug for cell in config.all_cells}) == 12
    assert config.recovery_thresholds == {
        "accuracy": 0.55,
        "target_recovery_ratio": 0.50,
        "control_retention_ratio": 0.80,
    }
    assert config.bootstrap_draws == 10_000
    assert config.bootstrap_seed == 20260830
    assert config.disk_reserve_gib == 8.0
    assert config.base_run == Path(
        "results/reviewer_revision_2026_08/20260830T024520Z-7aab3eb"
    )
    assert config.source_path == SPEC_PATH.resolve()


def test_construct_cells_are_orderable_frozen_and_have_path_safe_slugs() -> None:
    later = ConstructCell("qwen", "Qwen/Qwen2.5-1.5B", "sst2", 14)
    earlier = ConstructCell("bert", "google-bert/bert-base-uncased", "sva", 6)

    assert sorted((later, earlier)) == [earlier, later]
    assert later.slug == "qwen-sst2-l14"
    with pytest.raises(FrozenInstanceError):
        later.layer = 8  # type: ignore[misc]


def test_extension_config_and_nested_thresholds_are_immutable() -> None:
    config = load_extension_config(SPEC_PATH)

    with pytest.raises(FrozenInstanceError):
        config.bootstrap_draws = 1  # type: ignore[misc]
    with pytest.raises(TypeError):
        config.recovery_thresholds["accuracy"] = 0.0  # type: ignore[index]


def test_extension_config_record_is_portable_json_and_detached() -> None:
    config = load_extension_config(SPEC_PATH)

    record = config.to_record()
    expected = {
        "config_hash": config.config_hash,
        "base_run": (
            "results/reviewer_revision_2026_08/20260830T024520Z-7aab3eb"
        ),
        "pilot": {
            "model_key": PILOT.model_key,
            "model_id": PILOT.model_id,
            "task": PILOT.task,
            "layer": PILOT.layer,
        },
        "confirmatory_cells": [
            {
                "model_key": cell.model_key,
                "model_id": cell.model_id,
                "task": cell.task,
                "layer": cell.layer,
            }
            for cell in CONFIRMATORY_CELLS
        ],
        "recovery_thresholds": {
            "accuracy": 0.55,
            "target_recovery_ratio": 0.50,
            "control_retention_ratio": 0.80,
        },
        "bootstrap_draws": 10_000,
        "bootstrap_seed": 20260830,
        "disk_reserve_gib": 8.0,
    }

    assert record == expected
    assert "source_path" not in record
    assert json.loads(json.dumps(record, sort_keys=True)) == record
    assert re.fullmatch(r"[0-9a-f]{64}", sha256_json(record))

    record["pilot"]["model_key"] = "mutated"
    record["confirmatory_cells"][0]["layer"] = 999
    record["confirmatory_cells"].append(record["pilot"])
    record["recovery_thresholds"]["accuracy"] = 0.0

    assert config.pilot == PILOT
    assert config.confirmatory_cells == CONFIRMATORY_CELLS
    assert config.recovery_thresholds["accuracy"] == 0.55
    assert config.to_record() == expected


def test_extension_config_hash_uses_repository_canonical_json_convention(
    tmp_path: Path,
) -> None:
    payload = _valid_extension_payload()
    reordered_path = tmp_path / "reordered.yaml"
    reordered_path.write_text(
        "# Different formatting and mapping order must not change the hash.\n"
        + yaml.safe_dump(payload, sort_keys=True),
        encoding="utf-8",
    )

    original = load_extension_config(SPEC_PATH)
    reordered = load_extension_config(reordered_path)

    assert original.config_hash == reordered.config_hash == sha256_json(payload)
    assert re.fullmatch(r"[0-9a-f]{64}", original.config_hash)


def test_extension_config_rejects_pilot_in_confirmatory_cells(tmp_path: Path) -> None:
    payload = _valid_extension_payload()
    payload["construct_panel"]["confirmatory_cells"].append(
        payload["construct_panel"]["pilot"]
    )

    with pytest.raises(ExtensionConfigError, match="pilot"):
        load_extension_config(_write_yaml(tmp_path, payload))


def test_extension_config_rejects_duplicate_confirmatory_cells(tmp_path: Path) -> None:
    payload = _valid_extension_payload()
    cells = payload["construct_panel"]["confirmatory_cells"]
    cells[1] = cells[0].copy()

    with pytest.raises(ExtensionConfigError, match="11 unique confirmatory"):
        load_extension_config(_write_yaml(tmp_path, payload))


def test_extension_config_rejects_duplicate_cell_slugs(tmp_path: Path) -> None:
    payload = _valid_extension_payload()
    cells = payload["construct_panel"]["confirmatory_cells"]
    cells[1] = {**cells[0], "model_id": "different/model-id"}

    with pytest.raises(ExtensionConfigError, match="duplicate.*slug"):
        load_extension_config(_write_yaml(tmp_path, payload))


def test_extension_config_rejects_non_panel_middle_layer_cell(tmp_path: Path) -> None:
    payload = _valid_extension_payload()
    payload["construct_panel"]["confirmatory_cells"][0]["layer"] = 4

    with pytest.raises(ExtensionConfigError, match="locked 12-cell middle-layer panel"):
        load_extension_config(_write_yaml(tmp_path, payload))


@pytest.mark.parametrize(
    "threshold",
    ("accuracy", "target_recovery_ratio", "control_retention_ratio"),
)
def test_extension_config_rejects_recovery_threshold_drift(
    tmp_path: Path, threshold: str
) -> None:
    payload = _valid_extension_payload()
    payload["construct_panel"]["recovery_thresholds"][threshold] += 0.01

    with pytest.raises(ExtensionConfigError, match="recovery thresholds"):
        load_extension_config(_write_yaml(tmp_path, payload))


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    (
        (("inference", "bootstrap", "draws"), 9_999, "10,000"),
        (("inference", "bootstrap", "seed"), 17, "bootstrap.seed"),
        (("inference", "alternative"), "two_sided", "one_sided"),
        (("inference", "multiplicity"), "bonferroni", "Holm"),
        (
            ("inference", "bootstrap", "shared_resamples_across_models_within_task"),
            False,
            "shared_resamples",
        ),
    ),
)
def test_extension_config_rejects_bootstrap_or_inference_drift(
    tmp_path: Path,
    path: tuple[str | int, ...],
    replacement: Any,
    message: str,
) -> None:
    payload = _valid_extension_payload()
    _set_path(payload, path, replacement)

    with pytest.raises(ExtensionConfigError, match=message):
        load_extension_config(_write_yaml(tmp_path, payload))


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    (
        (("storage", "disk_reserve_gib"), 7.5, "8 GiB"),
        (
            ("outputs", "root"),
            "results/reviewer_caveat_extension_other",
            "outputs.root",
        ),
        (("outputs", "immutable_run_directory"), False, "immutable"),
    ),
)
def test_extension_config_rejects_storage_or_output_drift(
    tmp_path: Path,
    path: tuple[str | int, ...],
    replacement: Any,
    message: str,
) -> None:
    payload = _valid_extension_payload()
    _set_path(payload, path, replacement)

    with pytest.raises(ExtensionConfigError, match=message):
        load_extension_config(_write_yaml(tmp_path, payload))


@pytest.mark.parametrize(
    "base_run",
    (
        "../reviewer_revision_2026_08/20260830T024520Z-7aab3eb",
        "/tmp/reviewer_revision_2026_08/20260830T024520Z-7aab3eb",
        r"results\reviewer_revision_2026_08\..\escape",
    ),
)
def test_extension_config_rejects_base_run_path_escape(
    tmp_path: Path, base_run: str
) -> None:
    payload = _valid_extension_payload()
    payload["base_run"] = base_run

    with pytest.raises(ExtensionConfigError, match="base_run.*relative path"):
        load_extension_config(_write_yaml(tmp_path, payload))


def test_extension_config_rejects_path_unsafe_cell_slug(tmp_path: Path) -> None:
    payload = _valid_extension_payload()
    payload["construct_panel"]["confirmatory_cells"][0]["model_key"] = "../pythia"

    with pytest.raises(ExtensionConfigError, match="model_key.*path-safe"):
        load_extension_config(_write_yaml(tmp_path, payload))


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "missing.*inference"),
        ("unknown", "unexpected.*unregistered_option"),
        ("wrong_sequence_type", "confirmatory_cells.*sequence"),
        ("wrong_layer_type", "layer.*integer"),
        ("boolean_draws", "draws.*integer"),
    ),
)
def test_extension_config_rejects_missing_unknown_or_malformed_fields(
    tmp_path: Path, mutation: str, message: str
) -> None:
    payload = _valid_extension_payload()
    if mutation == "missing":
        del payload["inference"]
    elif mutation == "unknown":
        payload["unregistered_option"] = True
    elif mutation == "wrong_sequence_type":
        payload["construct_panel"]["confirmatory_cells"] = "not-a-sequence"
    elif mutation == "wrong_layer_type":
        payload["construct_panel"]["confirmatory_cells"][0]["layer"] = "6"
    elif mutation == "boolean_draws":
        payload["inference"]["bootstrap"]["draws"] = True
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(mutation)

    with pytest.raises(ExtensionConfigError, match=message):
        load_extension_config(_write_yaml(tmp_path, payload))


def test_extension_config_rejects_non_mapping_root(tmp_path: Path) -> None:
    with pytest.raises(ExtensionConfigError, match="root.*mapping"):
        load_extension_config(_write_yaml(tmp_path, ["not", "a", "mapping"]))


def test_extension_config_rejects_duplicate_yaml_mapping_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text(
        SPEC_PATH.read_text(encoding="utf-8") + "\nschema_version: 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ExtensionConfigError, match="duplicate.*schema_version"):
        load_extension_config(path)
