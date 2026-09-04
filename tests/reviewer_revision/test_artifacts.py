from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.reviewer_revision.artifacts import (
    ArtifactValidationError,
    RunContext,
    RunLockedError,
    assert_exact_keys,
    atomic_save_numpy,
    atomic_torch_save,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_via,
    canonical_json,
    sha256_file,
    sha256_json,
    sha256_tensor,
)
from src.reviewer_revision.config import EpsilonSweepRowKey


def test_canonical_json_and_hashes_are_stable(tmp_path: Path) -> None:
    left = {"z": [3, 2, 1], "a": {"b": True}, "path": Path("relative/file.txt")}
    right = {"path": "relative/file.txt", "a": {"b": True}, "z": [3, 2, 1]}

    assert canonical_json(left) == canonical_json(right)
    assert sha256_json(left) == sha256_json(right)

    file_path = tmp_path / "payload.bin"
    file_path.write_bytes(b"reviewer-revision")
    assert sha256_file(file_path) == "9d0f1f367762ec2d15cccff228151ebcce30d041e634498457a661f1c04b5b9d"

    tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    assert sha256_tensor(tensor) == sha256_tensor(tensor.clone())
    assert sha256_tensor(tensor) != sha256_tensor(tensor.to(torch.float64))

    with pytest.raises(ValueError, match="finite"):
        canonical_json({"bad": float("nan")})


def test_interrupted_atomic_write_leaves_no_final_or_temp_file(tmp_path: Path) -> None:
    destination = tmp_path / "shard.json"

    def interrupt(temp_path: Path) -> None:
        temp_path.write_text("partial", encoding="utf-8")
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        atomic_write_via(destination, interrupt)

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_all_supported_artifact_types_are_written_atomically(tmp_path: Path) -> None:
    json_path = tmp_path / "row.json"
    csv_path = tmp_path / "rows.csv"
    parquet_path = tmp_path / "rows.parquet"
    torch_path = tmp_path / "checkpoint.pt"
    numpy_path = tmp_path / "array.npy"

    atomic_write_json(json_path, {"value": 7})
    atomic_write_csv(csv_path, [{"key": "a", "value": 1}, {"key": "b", "value": 2}])
    atomic_write_parquet(parquet_path, pd.DataFrame([{"key": "a", "value": 1}]))
    atomic_torch_save(torch_path, {"weight": torch.tensor([1.0, 2.0])})
    atomic_save_numpy(numpy_path, np.array([3.0, 4.0], dtype=np.float32))

    assert json.loads(json_path.read_text(encoding="utf-8")) == {"value": 7}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == [
            {"key": "a", "value": "1"},
            {"key": "b", "value": "2"},
        ]
    assert pd.read_parquet(parquet_path).to_dict("records") == [{"key": "a", "value": 1}]
    assert torch.equal(torch.load(torch_path, weights_only=True)["weight"], torch.tensor([1.0, 2.0]))
    assert np.array_equal(np.load(numpy_path), np.array([3.0, 4.0], dtype=np.float32))
    assert not list(tmp_path.glob("*.tmp"))


def test_run_context_is_exclusive_immutable_and_sanitizes_manifest_paths(tmp_path: Path) -> None:
    fixed_time = datetime(2026, 8, 29, 12, 34, 56, tzinfo=timezone.utc)
    secret_path = tmp_path / "Users" / "local-user" / "private" / "cache.pt"

    with RunContext.create(
        output_root=tmp_path / "runs",
        config_hash="c" * 64,
        git_commit="163c1382e5747be4e59b987c9e0dc9c82d596b1a",
        timestamp=fixed_time,
        manifest={"source_cache": secret_path, "nested": {"home": str(secret_path.parent)}},
    ) as run:
        assert run.run_id == "20260829T123456Z-163c138"
        assert run.lock_path.exists()
        assert run.console_log_path.exists()
        assert run.shards_dir.is_dir()
        manifest_text = run.manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(manifest_text)
        assert run.manifest_path.name == "run_manifest.json"
        assert str(tmp_path) not in manifest_text
        assert "local-user" not in manifest_text
        assert manifest["source_cache"] == "<ABS_PATH>/cache.pt"
        assert manifest["nested"]["home"] == "<ABS_PATH>/private"

        with pytest.raises(RunLockedError):
            RunContext.resume(run.run_dir, config_hash="c" * 64)

    assert not run.lock_path.exists()
    with pytest.raises(FileExistsError):
        RunContext.create(
            output_root=tmp_path / "runs",
            config_hash="c" * 64,
            git_commit="163c1382e5747be4e59b987c9e0dc9c82d596b1a",
            timestamp=fixed_time,
        )


def test_resume_requires_matching_manifest_and_exact_validated_shard_keys(tmp_path: Path) -> None:
    key = EpsilonSweepRowKey("qwen", "sst2", 14, 0, "pgd", 0.5, "split")
    fixed_time = datetime(2026, 8, 29, tzinfo=timezone.utc)

    with RunContext.create(
        output_root=tmp_path,
        config_hash="a" * 64,
        git_commit="b" * 40,
        timestamp=fixed_time,
    ) as run:
        assert run.write_json_shard(
            "epsilon_sweep",
            key,
            {"status": "ok", "C": 1.0},
            validator=lambda payload: payload["status"] in {"ok", "failed"},
        )
        assert run.load_json_shard("epsilon_sweep", key) == {"status": "ok", "C": 1.0}
        assert not run.write_json_shard(
            "epsilon_sweep",
            key,
            {"status": "ok", "C": 999.0},
        )
        run.update_manifest({"stage": "experiment_b", "local_path": tmp_path / "secret.txt"})
        updated_manifest_text = run.manifest_path.read_text(encoding="utf-8")
        assert str(tmp_path) not in updated_manifest_text
        assert json.loads(updated_manifest_text)["local_path"] == "<ABS_PATH>/secret.txt"
        run_dir = run.run_dir

    with pytest.raises(ArtifactValidationError, match="config_hash"):
        RunContext.resume(run_dir, config_hash="d" * 64)

    with RunContext.resume(run_dir, config_hash="a" * 64, git_commit="b" * 40) as resumed:
        assert resumed.completed_keys("epsilon_sweep", expected_keys={key}) == {tuple(key)}
        with pytest.raises(ArtifactValidationError, match="unexpected"):
            resumed.completed_keys(
                "epsilon_sweep",
                expected_keys={EpsilonSweepRowKey("qwen", "sst2", 14, 1, "pgd", 0.5, "split")},
            )

        shard_path = resumed.shard_path("epsilon_sweep", key)
        envelope = json.loads(shard_path.read_text(encoding="utf-8"))
        envelope["payload"]["C"] = 0.0
        shard_path.write_text(json.dumps(envelope), encoding="utf-8")
        with pytest.raises(ArtifactValidationError, match="payload hash"):
            resumed.completed_keys("epsilon_sweep")


def test_assert_exact_keys_rejects_duplicates_missing_and_unexpected() -> None:
    expected = [("a", 1), ("b", 2)]

    assert assert_exact_keys(expected, list(reversed(expected))) == set(expected)
    with pytest.raises(ArtifactValidationError, match="duplicate"):
        assert_exact_keys(expected, [("a", 1), ("a", 1), ("b", 2)])
    with pytest.raises(ArtifactValidationError, match="missing"):
        assert_exact_keys(expected, [("a", 1)])
    with pytest.raises(ArtifactValidationError, match="unexpected"):
        assert_exact_keys(expected, [("a", 1), ("b", 2), ("c", 3)])


def test_failed_shard_is_retryable_and_preserved_when_retry_succeeds(tmp_path: Path) -> None:
    key = ("qwen", "sst2", 14, 0, "fgsm", "matched")
    with RunContext.create(
        output_root=tmp_path,
        config_hash="a" * 64,
        git_commit="b" * 40,
        timestamp=datetime(2026, 8, 29, 13, tzinfo=timezone.utc),
    ) as run:
        assert run.write_json_shard(
            "matched_split",
            key,
            {"status": "failed", "failure_reason": "transient interruption"},
        )
        assert run.completed_keys("matched_split", expected_keys={key}) == set()
        assert run.observed_keys("matched_split", expected_keys={key}) == {key}

        assert run.write_json_shard(
            "matched_split",
            key,
            {"status": "ok", "C": 0.25},
        )
        assert run.completed_keys("matched_split", expected_keys={key}) == {key}
        assert run.observed_keys("matched_split", expected_keys={key}) == {key}
        assert run.load_json_shard("matched_split", key) == {"status": "ok", "C": 0.25}

        archived = list((run.run_dir / "failed_shards" / "matched_split").glob("*.json"))
        assert len(archived) == 1
        envelope = json.loads(archived[0].read_text(encoding="utf-8"))
        assert envelope["payload"]["status"] == "failed"
        assert envelope["payload"]["failure_reason"] == "transient interruption"
