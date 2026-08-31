"""Machine-neutral environment and artifact manifests for reviewer reproduction."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from .artifacts import sha256_file, sha256_json

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"[0-9a-f]{7,64}\Z")
_TIER = re.compile(r"[a-z0-9][a-z0-9_.-]*\Z")
_PLATFORM_VALUE = re.compile(r"[A-Za-z0-9_. ()-]+\Z")
_PYTHON_VERSION = re.compile(r"[0-9]+(?:\.[0-9]+){1,3}(?:[A-Za-z0-9_.+-]*)?\Z")


def installed_distributions() -> list[dict[str, Any]]:
    """Return only dependency fields safe to inspect for a portable lock."""

    records: list[dict[str, Any]] = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name") or getattr(distribution, "name", None)
        direct_url: str | None = None
        editable = False
        direct_url_text = distribution.read_text("direct_url.json")
        if direct_url_text:
            try:
                payload = json.loads(direct_url_text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"distribution {name!r} has malformed direct URL metadata"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"distribution {name!r} has malformed direct URL metadata"
                )
            url = payload.get("url")
            if url is not None and not isinstance(url, str):
                raise ValueError(
                    f"distribution {name!r} has malformed direct URL metadata"
                )
            direct_url = url
            directory = payload.get("dir_info")
            if directory is not None:
                if not isinstance(directory, dict):
                    raise ValueError(
                        f"distribution {name!r} has malformed direct URL metadata"
                    )
                editable_value = directory.get("editable", False)
                if type(editable_value) is not bool:
                    raise ValueError(
                        f"distribution {name!r} has malformed editable metadata"
                    )
                editable = editable_value
        records.append(
            {
                "name": name,
                "version": distribution.version,
                "direct_url": direct_url,
                "editable": editable,
            }
        )
    return records


def _require_digest(value: Any, *, field: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{field} is not a valid lowercase hexadecimal digest")
    return value


def _clean_distribution(record: Any) -> dict[str, str]:
    if not isinstance(record, Mapping):
        raise TypeError("installed distribution records must be mappings")
    name = record.get("name")
    version = record.get("version")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("installed distribution name must be non-empty")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"distribution {name!r} has no version")
    if record.get("direct_url") not in (None, ""):
        raise ValueError(f"distribution {name!r} uses a direct URL and is not portable")
    editable = record.get("editable", False)
    if type(editable) is not bool:
        raise ValueError(f"distribution {name!r} has invalid editable metadata")
    if editable:
        raise ValueError(f"distribution {name!r} is editable and is not portable")
    try:
        parsed_version = Version(version)
    except InvalidVersion as exc:
        raise ValueError(f"distribution {name!r} has an invalid version") from exc
    if parsed_version != Version(str(parsed_version)):
        raise AssertionError("packaging version normalization is not idempotent")
    canonical_name = canonicalize_name(name)
    if not canonical_name or any(
        token in canonical_name for token in ("/", "\\", "@", ":")
    ):
        raise ValueError(f"distribution {name!r} has a nonportable name")
    return {"name": canonical_name, "version": version}


def build_environment_lock(
    *,
    spec_hash: str,
    git_commit: str,
    python_version: str | None = None,
    platform_record: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a reproducible dependency record without local paths or URLs."""

    _require_digest(spec_hash, field="spec_hash", pattern=_SHA256)
    _require_digest(git_commit, field="git_commit", pattern=_GIT_COMMIT)
    selected_python = python_version or platform.python_version()
    if (
        not isinstance(selected_python, str)
        or _PYTHON_VERSION.fullmatch(selected_python) is None
    ):
        raise ValueError("python_version is not a portable version string")
    selected_platform = dict(
        platform_record
        or {
            "system": platform.system(),
            "machine": platform.machine(),
        }
    )
    if set(selected_platform) != {"system", "machine"}:
        raise ValueError("platform record must contain exactly system and machine")
    for field, value in selected_platform.items():
        if (
            not isinstance(value, str)
            or not value
            or _PLATFORM_VALUE.fullmatch(value) is None
        ):
            raise ValueError(f"platform {field} is not machine-neutral")

    distributions = [
        _clean_distribution(record) for record in installed_distributions()
    ]
    distributions.sort(key=lambda record: (record["name"], record["version"]))
    names = [record["name"] for record in distributions]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"duplicate distribution names after normalization: {duplicates}"
        )
    requirements = [
        f"{record['name']}=={record['version']}" for record in distributions
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "spec_hash": spec_hash,
        "git_commit": git_commit,
        "python_version": selected_python,
        "python_implementation": sys.implementation.name,
        "platform": selected_platform,
        "distributions": distributions,
        "requirements": requirements,
    }
    payload["aggregate_sha256"] = sha256_json(payload)
    return payload


def _portable_relative(path: Path, root: Path) -> str:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"artifact escapes artifact root: {path}")
    relative = resolved.relative_to(resolved_root).as_posix()
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != relative
        or "\\" in relative
    ):
        raise ValueError(f"artifact path is not a portable relative path: {relative!r}")
    return relative


def build_artifact_manifest(
    root: str | Path,
    tiers: Mapping[str, Sequence[str | Path]],
) -> dict[str, Any]:
    """Hash nonempty files and record their union of reproduction tiers."""

    artifact_root = Path(root)
    if not artifact_root.is_dir():
        raise ValueError("artifact root must be an existing directory")
    if not isinstance(tiers, Mapping) or not tiers:
        raise ValueError("artifact tiers must be a non-empty mapping")
    memberships: dict[str, set[str]] = {}
    for tier, paths in tiers.items():
        if not isinstance(tier, str) or _TIER.fullmatch(tier) is None:
            raise ValueError(f"invalid artifact tier: {tier!r}")
        if isinstance(paths, (str, bytes, bytearray)) or not isinstance(
            paths, Sequence
        ):
            raise TypeError(f"artifact tier {tier!r} must contain a path sequence")
        raw_resolved = [Path(path).resolve() for path in paths]
        if len(raw_resolved) != len(set(raw_resolved)):
            raise ValueError(f"duplicate artifact in tier {tier!r}")
        for path in paths:
            candidate = Path(path)
            relative = _portable_relative(candidate, artifact_root)
            memberships.setdefault(relative, set()).add(tier)

    records: list[dict[str, Any]] = []
    for relative, member_tiers in sorted(memberships.items()):
        path = artifact_root / PurePosixPath(relative)
        if not path.is_file():
            raise ValueError(f"missing artifact: {relative}")
        size = path.stat().st_size
        if size <= 0:
            raise ValueError(f"empty artifact: {relative}")
        records.append(
            {
                "path": relative,
                "bytes": size,
                "sha256": sha256_file(path),
                "tiers": sorted(member_tiers),
            }
        )
    manifest = {
        "schema_version": 1,
        "files": records,
        "aggregate_sha256": sha256_json(records),
    }
    return manifest


def verify_artifact_manifest(
    root: str | Path,
    manifest: Mapping[str, Any],
) -> None:
    """Fail closed when a portable artifact manifest or any byte changes."""

    artifact_root = Path(root)
    if not artifact_root.is_dir():
        raise ValueError("artifact root must be an existing directory")
    if not isinstance(manifest, Mapping) or set(manifest) != {
        "schema_version",
        "files",
        "aggregate_sha256",
    }:
        raise ValueError("artifact manifest schema fields are invalid")
    if manifest.get("schema_version") != 1:
        raise ValueError("artifact manifest schema_version mismatch")
    records = manifest.get("files")
    if not isinstance(records, list):
        raise TypeError("artifact manifest files must be a list")
    expected_aggregate = manifest.get("aggregate_sha256")
    _require_digest(
        expected_aggregate,
        field="aggregate_sha256",
        pattern=_SHA256,
    )
    if sha256_json(records) != expected_aggregate:
        raise ValueError("artifact manifest aggregate hash mismatch")

    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "bytes",
            "sha256",
            "tiers",
        }:
            raise ValueError("artifact file record schema is invalid")
        relative = record["path"]
        if not isinstance(relative, str):
            raise TypeError("artifact path is not a portable relative path")
        pure = PurePosixPath(relative)
        if (
            not relative
            or pure.is_absolute()
            or pure.as_posix() != relative
            or "\\" in relative
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise ValueError(
                f"artifact path is not a portable relative path: {relative!r}"
            )
        if relative in seen:
            raise ValueError(f"duplicate artifact path: {relative}")
        seen.add(relative)
        size = record["bytes"]
        if type(size) is not int or size <= 0:
            raise ValueError(f"artifact size is invalid: {relative}")
        _require_digest(
            record["sha256"], field=f"sha256 for {relative}", pattern=_SHA256
        )
        tiers = record["tiers"]
        if (
            not isinstance(tiers, list)
            or not tiers
            or tiers != sorted(set(tiers))
            or any(
                not isinstance(tier, str) or _TIER.fullmatch(tier) is None
                for tier in tiers
            )
        ):
            raise ValueError(f"artifact tiers are invalid: {relative}")
        path = artifact_root / pure
        observed_relative = _portable_relative(path, artifact_root)
        if observed_relative != relative:
            raise ValueError(f"artifact path canonicalization mismatch: {relative}")
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"artifact size mismatch: {relative}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"artifact hash mismatch: {relative}")
