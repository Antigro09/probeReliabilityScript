"""Immutable, resume-safe artifact helpers for the reviewer revision run."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import os
import re
import tempfile
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import TracebackType
from typing import Any, Self


class ArtifactValidationError(ValueError):
    """Raised when an artifact cannot be trusted for analysis or resume."""


class RunLockedError(RuntimeError):
    """Raised when another process owns an immutable run directory."""


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_value(dataclasses.asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "_fields") and isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool)):
                raise TypeError(f"JSON object key is not scalar: {type(key).__name__}")
            text_key = str(key)
            if text_key in normalized:
                raise ValueError(f"duplicate canonical JSON key: {text_key!r}")
            normalized[text_key] = _json_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_json_value(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON requires finite floating-point values")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return _json_value(value.tolist())
        if isinstance(value, np.generic):
            return _json_value(value.item())
    except ImportError:  # pragma: no cover - NumPy is a declared dependency.
        pass

    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json(payload: Any) -> str:
    """Serialize *payload* deterministically, rejecting NaN and infinities."""

    return json.dumps(
        _json_value(payload),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(payload: Any) -> str:
    """Return the SHA-256 of the canonical JSON representation of *payload*."""

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor: Any) -> str:
    """Hash tensor dtype, shape, and exact contiguous CPU bytes."""

    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("sha256_tensor expects a torch.Tensor")
    if tensor.layout != torch.strided:
        raise TypeError("sha256_tensor requires a strided tensor")
    detached = tensor.detach().cpu().contiguous()
    byte_view = detached.reshape(-1).view(torch.uint8)
    metadata = canonical_json(
        {"dtype": str(detached.dtype), "shape": list(detached.shape)}
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(metadata)
    digest.update(b"\0")
    digest.update(byte_view.numpy().tobytes(order="C"))
    return digest.hexdigest()


def atomic_write_via(path: str | Path, writer: Callable[[Path], None]) -> Path:
    """Write to a same-directory temporary file, then atomically replace *path*."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temp_name)
    try:
        writer(temporary)
        # Windows requires a writable descriptor for ``fsync``.
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """Atomically write canonical UTF-8 JSON followed by a newline."""

    serialized = canonical_json(payload) + "\n"

    def write(temporary: Path) -> None:
        temporary.write_text(serialized, encoding="utf-8", newline="\n")

    return atomic_write_via(path, write)


def atomic_write_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Atomically write mapping rows as reviewer-readable CSV."""

    materialized = [dict(row) for row in rows]
    if fieldnames is None:
        columns: list[str] = []
        for row in materialized:
            for name in row:
                if name not in columns:
                    columns.append(name)
        fieldnames = columns
    names = list(fieldnames)

    def write(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=names, extrasaction="raise")
            writer.writeheader()
            writer.writerows(materialized)
            handle.flush()
            os.fsync(handle.fileno())

    return atomic_write_via(path, write)


def atomic_write_parquet(path: str | Path, frame_or_rows: Any) -> Path:
    """Atomically write a pandas DataFrame or iterable of mappings as Parquet."""

    import pandas as pd

    frame = frame_or_rows if isinstance(frame_or_rows, pd.DataFrame) else pd.DataFrame(frame_or_rows)

    def write(temporary: Path) -> None:
        frame.to_parquet(temporary, index=False)

    return atomic_write_via(path, write)


def atomic_torch_save(path: str | Path, payload: Any) -> Path:
    """Atomically persist a Torch checkpoint."""

    import torch

    return atomic_write_via(path, lambda temporary: torch.save(payload, temporary))


def atomic_save_numpy(path: str | Path, array: Any) -> Path:
    """Atomically persist an exact NumPy ``.npy`` array."""

    import numpy as np

    def write(temporary: Path) -> None:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())

    return atomic_write_via(path, write)


_WINDOWS_ABSOLUTE_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")


def _sanitize_path_string(value: str) -> str:
    if _WINDOWS_ABSOLUTE_RE.match(value):
        name = PureWindowsPath(value).name
        return f"<ABS_PATH>/{name}" if name else "<ABS_PATH>"
    if value.startswith("/"):
        name = PurePosixPath(value).name
        return f"<ABS_PATH>/{name}" if name else "<ABS_PATH>"
    return value


def sanitize_manifest_payload(payload: Any) -> Any:
    """Recursively replace absolute paths with machine-neutral leaf references."""

    if isinstance(payload, Path):
        return _sanitize_path_string(str(payload.resolve()))
    if isinstance(payload, str):
        return _sanitize_path_string(payload)
    if dataclasses.is_dataclass(payload) and not isinstance(payload, type):
        return sanitize_manifest_payload(dataclasses.asdict(payload))
    if isinstance(payload, Mapping):
        return {str(key): sanitize_manifest_payload(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [sanitize_manifest_payload(value) for value in payload]
    if isinstance(payload, (set, frozenset)):
        return sorted(sanitize_manifest_payload(value) for value in payload)
    return payload


def _key_tuple(key: Any) -> tuple[Any, ...]:
    if isinstance(key, tuple):
        values = tuple(key)
    elif dataclasses.is_dataclass(key) and not isinstance(key, type):
        values = tuple(dataclasses.asdict(key).values())
    elif isinstance(key, Sequence) and not isinstance(key, (str, bytes, bytearray)):
        values = tuple(key)
    else:
        raise TypeError("artifact keys must be tuples, named tuples, dataclasses, or sequences")
    normalized = _json_value(values)
    if not isinstance(normalized, list):  # pragma: no cover - guaranteed by tuple input.
        raise TypeError("artifact key did not normalize to a sequence")
    if any(isinstance(value, (list, dict)) for value in normalized):
        raise TypeError("artifact keys must contain only scalar values")
    return tuple(normalized)


def assert_exact_keys(expected: Iterable[Any], observed: Iterable[Any]) -> set[tuple[Any, ...]]:
    """Require observed keys to be unique and exactly equal to the requested keys."""

    expected_list = [_key_tuple(key) for key in expected]
    observed_list = [_key_tuple(key) for key in observed]
    expected_counts = Counter(expected_list)
    observed_counts = Counter(observed_list)
    duplicate_expected = sorted(key for key, count in expected_counts.items() if count > 1)
    duplicate_observed = sorted(key for key, count in observed_counts.items() if count > 1)
    if duplicate_expected:
        raise ArtifactValidationError(f"duplicate expected keys: {duplicate_expected!r}")
    if duplicate_observed:
        raise ArtifactValidationError(f"duplicate observed keys: {duplicate_observed!r}")
    expected_set = set(expected_list)
    observed_set = set(observed_list)
    missing = sorted(expected_set - observed_set)
    unexpected = sorted(observed_set - expected_set)
    if missing:
        raise ArtifactValidationError(f"missing required keys: {missing!r}")
    if unexpected:
        raise ArtifactValidationError(f"unexpected keys: {unexpected!r}")
    return observed_set


def validate_json_shard(
    path: str | Path,
    *,
    expected_experiment: str | None = None,
    expected_key: Any | None = None,
) -> dict[str, Any]:
    """Load and validate a JSON shard envelope and its content hashes."""

    shard_path = Path(path)
    try:
        envelope = json.loads(shard_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"unreadable shard {shard_path.name}: {exc}") from exc
    if not isinstance(envelope, dict) or envelope.get("schema_version") != 1:
        raise ArtifactValidationError(f"invalid shard schema: {shard_path.name}")
    experiment = envelope.get("experiment")
    if not isinstance(experiment, str):
        raise ArtifactValidationError(f"invalid shard experiment: {shard_path.name}")
    if expected_experiment is not None and experiment != expected_experiment:
        raise ArtifactValidationError(
            f"shard experiment mismatch: expected {expected_experiment!r}, got {experiment!r}"
        )
    try:
        key = _key_tuple(envelope["key"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"invalid shard key: {shard_path.name}") from exc
    if expected_key is not None and key != _key_tuple(expected_key):
        raise ArtifactValidationError(f"shard key mismatch: {shard_path.name}")
    if envelope.get("key_sha256") != sha256_json(list(key)):
        raise ArtifactValidationError(f"shard key hash mismatch: {shard_path.name}")
    if "payload" not in envelope:
        raise ArtifactValidationError(f"shard payload missing: {shard_path.name}")
    if envelope.get("payload_sha256") != sha256_json(envelope["payload"]):
        raise ArtifactValidationError(f"shard payload hash mismatch: {shard_path.name}")
    return envelope


_EXPERIMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class RunContext:
    """Exclusive owner of one timestamped, immutable revision run directory."""

    def __init__(self, run_dir: Path, manifest: dict[str, Any], lock_token: str) -> None:
        self.run_dir = run_dir
        self.run_id = str(manifest["run_id"])
        self.manifest_path = run_dir / "run_manifest.json"
        self.console_log_path = run_dir / "console.log"
        self.shards_dir = run_dir / "shards"
        self.lock_path = run_dir / ".run.lock"
        self._manifest = manifest
        self._lock_token = lock_token
        self._closed = False

    @staticmethod
    def _acquire_lock(lock_path: Path) -> str:
        token = uuid.uuid4().hex
        payload = (canonical_json({"pid": os.getpid(), "token": token}) + "\n").encode("utf-8")
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RunLockedError(f"run is already locked: {lock_path.parent.name}") from exc
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return token

    @classmethod
    def create(
        cls,
        *,
        output_root: str | Path,
        config_hash: str,
        git_commit: str,
        timestamp: datetime | None = None,
        manifest: Mapping[str, Any] | None = None,
    ) -> RunContext:
        if not re.fullmatch(r"[0-9a-f]{64}", config_hash):
            raise ArtifactValidationError("config_hash must be a lowercase SHA-256")
        if not re.fullmatch(r"[0-9a-f]{7,64}", git_commit):
            raise ArtifactValidationError("git_commit must be a lowercase hexadecimal revision")
        when = timestamp or datetime.now(timezone.utc)
        if when.tzinfo is None or when.utcoffset() is None:
            raise ArtifactValidationError("run timestamp must be timezone-aware")
        utc = when.astimezone(timezone.utc).replace(microsecond=0)
        run_id = f"{utc.strftime('%Y%m%dT%H%M%SZ')}-{git_commit[:7]}"
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        run_dir = root / run_id
        run_dir.mkdir(exist_ok=False)
        shards_dir = run_dir / "shards"
        shards_dir.mkdir()
        (run_dir / "console.log").touch(exist_ok=False)
        lock_path = run_dir / ".run.lock"
        lock_token = cls._acquire_lock(lock_path)

        supplied = dict(manifest or {})
        reserved = {"schema_version", "run_id", "created_at_utc", "git_commit", "config_hash"}
        overlap = reserved.intersection(supplied)
        if overlap:
            lock_path.unlink(missing_ok=True)
            raise ArtifactValidationError(f"reserved manifest keys supplied: {sorted(overlap)!r}")
        manifest_payload = {
            "schema_version": 1,
            "run_id": run_id,
            "created_at_utc": utc.isoformat().replace("+00:00", "Z"),
            "git_commit": git_commit,
            "config_hash": config_hash,
            **sanitize_manifest_payload(supplied),
        }
        try:
            atomic_write_json(run_dir / "run_manifest.json", manifest_payload)
        except BaseException:
            lock_path.unlink(missing_ok=True)
            raise
        return cls(run_dir, manifest_payload, lock_token)

    @classmethod
    def resume(
        cls,
        run_dir: str | Path,
        *,
        config_hash: str,
        git_commit: str | None = None,
    ) -> RunContext:
        directory = Path(run_dir)
        manifest_path = directory / "run_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ArtifactValidationError(f"unreadable run_manifest.json: {exc}") from exc
        if manifest.get("schema_version") != 1:
            raise ArtifactValidationError("run manifest schema_version mismatch")
        if manifest.get("run_id") != directory.name:
            raise ArtifactValidationError("run manifest run_id does not match directory")
        if manifest.get("config_hash") != config_hash:
            raise ArtifactValidationError("run manifest config_hash mismatch")
        if git_commit is not None and manifest.get("git_commit") != git_commit:
            raise ArtifactValidationError("run manifest git_commit mismatch")
        if not (directory / "shards").is_dir() or not (directory / "console.log").is_file():
            raise ArtifactValidationError("run directory is missing required artifact structure")
        lock_token = cls._acquire_lock(directory / ".run.lock")
        return cls(directory, manifest, lock_token)

    def __enter__(self) -> Self:
        if self._closed:
            raise RuntimeError("cannot re-enter a closed RunContext")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        try:
            current = json.loads(self.lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = {}
        if current.get("token") == self._lock_token:
            self.lock_path.unlink(missing_ok=True)
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("RunContext is closed")

    @staticmethod
    def _validate_experiment_name(experiment: str) -> None:
        if not _EXPERIMENT_RE.fullmatch(experiment):
            raise ArtifactValidationError(f"invalid experiment shard name: {experiment!r}")

    def shard_path(self, experiment: str, key: Any) -> Path:
        self._validate_experiment_name(experiment)
        key_hash = sha256_json(list(_key_tuple(key)))
        return self.shards_dir / experiment / f"{key_hash}.json"

    def write_json_shard(
        self,
        experiment: str,
        key: Any,
        payload: Any,
        *,
        validator: Callable[[Any], bool | None] | None = None,
    ) -> bool:
        """Write one immutable keyed shard; return ``False`` when it already exists."""

        self._require_open()
        key_tuple = _key_tuple(key)
        shard_path = self.shard_path(experiment, key_tuple)
        if shard_path.exists():
            existing = validate_json_shard(
                shard_path, expected_experiment=experiment, expected_key=key_tuple
            )
            if not (
                isinstance(existing.get("payload"), Mapping)
                and existing["payload"].get("status") == "failed"
            ):
                return False
            failed_directory = self.run_dir / "failed_shards" / experiment
            failed_directory.mkdir(parents=True, exist_ok=True)
            archived_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            archived_path = failed_directory / (
                f"{shard_path.stem}.{archived_at}.{uuid.uuid4().hex[:8]}.json"
            )
            os.replace(shard_path, archived_path)
        if validator is not None and validator(payload) is False:
            raise ArtifactValidationError(f"payload validator rejected {experiment} shard")
        envelope = {
            "schema_version": 1,
            "experiment": experiment,
            "key": list(key_tuple),
            "key_sha256": sha256_json(list(key_tuple)),
            "payload": payload,
            "payload_sha256": sha256_json(payload),
        }
        atomic_write_json(shard_path, envelope)
        validate_json_shard(shard_path, expected_experiment=experiment, expected_key=key_tuple)
        return True

    def load_json_shard(self, experiment: str, key: Any) -> Any:
        self._require_open()
        envelope = validate_json_shard(
            self.shard_path(experiment, key),
            expected_experiment=experiment,
            expected_key=key,
        )
        return envelope["payload"]

    def _validated_shard_keys(
        self, experiment: str, *, expected_keys: Iterable[Any] | None = None
    ) -> tuple[set[tuple[Any, ...]], set[tuple[Any, ...]]]:
        """Return all and successful validated keys for an experiment."""

        self._require_open()
        self._validate_experiment_name(experiment)
        directory = self.shards_dir / experiment
        keys: list[tuple[Any, ...]] = []
        completed_keys: list[tuple[Any, ...]] = []
        if directory.exists():
            for shard_path in sorted(directory.glob("*.json")):
                envelope = validate_json_shard(shard_path, expected_experiment=experiment)
                key = _key_tuple(envelope["key"])
                keys.append(key)
                payload = envelope.get("payload")
                if not (
                    isinstance(payload, Mapping) and payload.get("status") == "failed"
                ):
                    completed_keys.append(key)
        counts = Counter(keys)
        duplicates = sorted(key for key, count in counts.items() if count > 1)
        if duplicates:
            raise ArtifactValidationError(f"duplicate completed shard keys: {duplicates!r}")
        completed = set(completed_keys)
        if expected_keys is not None:
            expected_list = [_key_tuple(key) for key in expected_keys]
            if len(expected_list) != len(set(expected_list)):
                raise ArtifactValidationError("duplicate expected resume keys")
            unexpected = sorted(set(keys) - set(expected_list))
            if unexpected:
                raise ArtifactValidationError(f"unexpected completed shard keys: {unexpected!r}")
        return set(keys), completed

    def observed_keys(
        self, experiment: str, *, expected_keys: Iterable[Any] | None = None
    ) -> set[tuple[Any, ...]]:
        """Return all validated shard keys, including explicit failure rows."""

        observed, _ = self._validated_shard_keys(
            experiment, expected_keys=expected_keys
        )
        return observed

    def completed_keys(
        self, experiment: str, *, expected_keys: Iterable[Any] | None = None
    ) -> set[tuple[Any, ...]]:
        """Return successful keys only, so explicit failures remain retryable."""

        _, completed = self._validated_shard_keys(
            experiment, expected_keys=expected_keys
        )
        return completed

    def update_manifest(self, updates: Mapping[str, Any]) -> None:
        """Atomically add run-stage metadata without exposing absolute paths."""

        self._require_open()
        reserved = {"schema_version", "run_id", "created_at_utc", "git_commit", "config_hash"}
        overlap = reserved.intersection(updates)
        if overlap:
            raise ArtifactValidationError(f"cannot update reserved manifest keys: {sorted(overlap)!r}")
        self._manifest = {**self._manifest, **sanitize_manifest_payload(dict(updates))}
        atomic_write_json(self.manifest_path, self._manifest)


__all__ = [
    "ArtifactValidationError",
    "RunContext",
    "RunLockedError",
    "assert_exact_keys",
    "atomic_save_numpy",
    "atomic_torch_save",
    "atomic_write_csv",
    "atomic_write_json",
    "atomic_write_parquet",
    "atomic_write_via",
    "canonical_json",
    "sanitize_manifest_payload",
    "sha256_file",
    "sha256_json",
    "sha256_tensor",
    "validate_json_shard",
]
