"""Machine-neutral environment and artifact manifests for reviewer reproduction."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import re
import sys
import sysconfig
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from packaging.utils import InvalidName, canonicalize_name
from packaging.version import InvalidVersion, Version

from .artifacts import sha256_file, sha256_json

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SAFE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_PLATFORM_VALUE = re.compile(r"[A-Za-z0-9_. ()+-]+\Z")
_PROTOCOL_VALUE = re.compile(r"[A-Za-z0-9_.:+-]+\Z")
_PYTHON_VERSION = re.compile(r"[0-9]+(?:\.[0-9]+){1,3}(?:[A-Za-z0-9_.+-]*)?\Z")
_WINDOWS_INVALID = frozenset('<>:"\\|?*')
_WINDOWS_RESERVED = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{number}" for number in range(1, 10)),
        *(f"LPT{number}" for number in range(1, 10)),
    }
)


def installed_distributions() -> list[dict[str, Any]]:
    """Return path-free PEP-610 source markers plus package names and versions."""

    records: list[dict[str, Any]] = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name") or getattr(distribution, "name", None)
        direct_url_text = distribution.read_text("direct_url.json")
        has_direct_url = direct_url_text is not None
        editable = False
        if has_direct_url:
            try:
                payload = json.loads(direct_url_text)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"distribution {name!r} has malformed direct-source metadata"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"distribution {name!r} has malformed direct-source metadata"
                )
            directory = payload.get("dir_info")
            if directory is not None:
                if not isinstance(directory, dict):
                    raise ValueError(
                        f"distribution {name!r} has malformed direct-source metadata"
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
                "has_direct_url": has_direct_url,
                "editable": editable,
            }
        )
    return records


def _require_digest(value: Any, *, field: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{field} is not a valid lowercase hexadecimal digest")
    return value


def _clean_distribution(record: Any) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError("installed distribution records must be mappings")
    name = record.get("name")
    version = record.get("version")
    if not isinstance(name, str) or not name or name != name.strip():
        raise ValueError(
            "installed distribution name must be canonicalizable and nonblank"
        )
    if not isinstance(version, str) or not version or version != version.strip():
        raise ValueError(f"distribution {name!r} has a noncanonical version")
    try:
        canonical_name = canonicalize_name(name, validate=True)
    except InvalidName as exc:
        raise ValueError(f"distribution {name!r} has an invalid name") from exc
    try:
        parsed_version = Version(version)
    except InvalidVersion as exc:
        raise ValueError(f"distribution {name!r} has an invalid version") from exc
    normalized_version = str(parsed_version)
    editable = record.get("editable", False)
    has_direct_url = record.get("has_direct_url", False)
    if type(editable) is not bool or type(has_direct_url) is not bool:
        raise ValueError(f"distribution {name!r} has invalid source metadata")
    if editable:
        raise ValueError(f"distribution {name!r} is editable and is not portable")
    if has_direct_url:
        source_kind = "direct_url_redacted"
        nonstandard_reason = "direct-source URL intentionally redacted"
    elif (
        parsed_version.is_devrelease
        or parsed_version.is_prerelease
        or parsed_version.local is not None
    ):
        source_kind = "nonstandard_build"
        nonstandard_reason = "development, prerelease, or local build"
    else:
        source_kind = "index"
        nonstandard_reason = None
    output: dict[str, Any] = {
        "name": canonical_name,
        "version": normalized_version,
        "source_kind": source_kind,
        "reconstructible_requirement": source_kind == "index",
    }
    if nonstandard_reason is not None:
        output["nonstandard_reason"] = nonstandard_reason
    return output


def _safe_platform_record(value: Mapping[str, str] | None) -> dict[str, str]:
    selected = (
        {"system": platform.system(), "machine": platform.machine()}
        if value is None
        else dict(value)
    )
    if set(selected) != {"system", "machine"}:
        raise ValueError("platform record must contain exactly system and machine")
    for field, item in selected.items():
        if (
            not isinstance(item, str)
            or not item
            or _PLATFORM_VALUE.fullmatch(item) is None
        ):
            raise ValueError(f"platform {field} is not machine-neutral")
    return selected


def _safe_device_protocol(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("device_protocol must be a non-empty mapping")
    output: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or _SAFE_TOKEN.fullmatch(key) is None:
            raise ValueError(f"invalid device-protocol key: {key!r}")
        if (
            not isinstance(item, str)
            or not item
            or _PROTOCOL_VALUE.fullmatch(item) is None
        ):
            raise ValueError(f"invalid device-protocol value for {key!r}")
        output[key] = item
    return dict(sorted(output.items()))


def _requirements_source(path: str | Path | None) -> dict[str, Any]:
    selected = _PROJECT_ROOT / "requirements.txt" if path is None else Path(path)
    if not selected.is_file() or selected.stat().st_size <= 0:
        raise ValueError("source requirements file is missing or empty")
    if selected.name != "requirements.txt":
        raise ValueError("source requirements file must be named requirements.txt")
    return {
        "path": selected.name,
        "bytes": selected.stat().st_size,
        "sha256": sha256_file(selected),
    }


def build_environment_lock(
    *,
    spec_hash: str,
    git_commit: str,
    producing_run: str,
    device_protocol: Mapping[str, str],
    deterministic_algorithms: bool,
    requirements_path: str | Path | None = None,
    python_version: str | None = None,
    python_abi: str | None = None,
    platform_record: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a sanitized lock that distinguishes reconstructible dependencies."""

    _require_digest(spec_hash, field="spec_hash", pattern=_SHA256)
    _require_digest(git_commit, field="git_commit", pattern=_GIT_COMMIT)
    if (
        not isinstance(producing_run, str)
        or _SAFE_TOKEN.fullmatch(producing_run) is None
    ):
        raise ValueError("producing_run must be a portable identifier")
    if type(deterministic_algorithms) is not bool:
        raise TypeError("deterministic_algorithms must be a Boolean")
    selected_python = (
        platform.python_version() if python_version is None else python_version
    )
    if (
        not isinstance(selected_python, str)
        or not selected_python
        or _PYTHON_VERSION.fullmatch(selected_python) is None
    ):
        raise ValueError("python_version is not a portable version string")
    selected_abi = (
        str(sysconfig.get_config_var("SOABI") or sys.implementation.cache_tag)
        if python_abi is None
        else python_abi
    )
    if (
        not isinstance(selected_abi, str)
        or not selected_abi
        or _PLATFORM_VALUE.fullmatch(selected_abi) is None
    ):
        raise ValueError("python_abi is not a portable ABI identifier")
    selected_platform = _safe_platform_record(platform_record)
    selected_protocol = _safe_device_protocol(device_protocol)

    distributions = [
        _clean_distribution(record) for record in installed_distributions()
    ]
    distributions.sort(key=lambda record: (record["name"], record["version"]))
    names = [record["name"] for record in distributions]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(
            f"duplicate distribution names after normalization: {duplicates}"
        )
    by_name = {record["name"]: record for record in distributions}
    if "torch" not in by_name:
        raise ValueError("the current environment does not contain PyTorch")
    requirements = [
        f"{record['name']}=={record['version']}"
        for record in distributions
        if record["reconstructible_requirement"]
    ]
    nonstandard = [
        dict(record)
        for record in distributions
        if not record["reconstructible_requirement"]
    ]
    torch_record = by_name["torch"]
    pytorch_build = {
        "version": torch_record["version"],
        "source_kind": torch_record["source_kind"],
        "reconstructible_requirement": torch_record["reconstructible_requirement"],
        "build_kind": (
            "standard_release"
            if torch_record["source_kind"] == "index"
            else "nonstandard_or_direct_source"
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "producing_run": producing_run,
        "spec_hash": spec_hash,
        "git_commit": git_commit,
        "python": {
            "version": selected_python,
            "implementation": sys.implementation.name,
            "abi": selected_abi,
        },
        "platform": selected_platform,
        "source_requirements": _requirements_source(requirements_path),
        "pytorch_build": pytorch_build,
        "device_protocol": selected_protocol,
        "deterministic_algorithms": deterministic_algorithms,
        "distributions": distributions,
        "requirements": requirements,
        "nonstandard_distributions": nonstandard,
    }
    payload["aggregate_sha256"] = sha256_json(payload)
    return payload


def _portable_path_parts(relative: str) -> tuple[str, ...]:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("artifact path is not a portable relative path")
    pure = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    if (
        pure.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or pure.as_posix() != relative
    ):
        raise ValueError(f"artifact path is not a portable relative path: {relative!r}")
    parts = pure.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"artifact path is not a portable relative path: {relative!r}")
    for part in parts:
        if unicodedata.normalize("NFC", part) != part:
            raise ValueError(f"artifact path is not Unicode-normalized: {relative!r}")
        if part.endswith((".", " ")):
            raise ValueError(f"artifact path is not portable to Windows: {relative!r}")
        if any(
            character in _WINDOWS_INVALID or ord(character) < 32 for character in part
        ):
            raise ValueError(f"artifact path is not portable to Windows: {relative!r}")
        reserved_stem = part.split(".", 1)[0].upper()
        if reserved_stem in _WINDOWS_RESERVED:
            raise ValueError(f"artifact path uses a Windows device name: {relative!r}")
    return parts


def _portable_relative(path: Path, root: Path) -> str:
    resolved_root = root.resolve()
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"artifact escapes artifact root: {path}")
    relative = resolved.relative_to(resolved_root).as_posix()
    _portable_path_parts(relative)
    return relative


def _collision_key(relative: str) -> str:
    parts = _portable_path_parts(relative)
    return "/".join(part.casefold() for part in parts)


def _require_producing_run(value: Any) -> str:
    if not isinstance(value, str) or _SAFE_TOKEN.fullmatch(value) is None:
        raise ValueError("producing_run must be a portable identifier")
    return value


def build_artifact_manifest(
    root: str | Path,
    tiers: Mapping[str, Sequence[str | Path]],
    *,
    producing_run: str,
) -> dict[str, Any]:
    """Hash nonempty files and record their reproduction tiers and producing run."""

    artifact_root = Path(root)
    if not artifact_root.is_dir():
        raise ValueError("artifact root must be an existing directory")
    artifact_root = artifact_root.resolve()
    producing_run = _require_producing_run(producing_run)
    if not isinstance(tiers, Mapping) or not tiers:
        raise ValueError("artifact tiers must be a non-empty mapping")
    memberships: dict[str, set[str]] = {}
    collision_paths: dict[str, str] = {}
    for tier, paths in tiers.items():
        if not isinstance(tier, str) or _SAFE_TOKEN.fullmatch(tier) is None:
            raise ValueError(f"invalid artifact tier: {tier!r}")
        if isinstance(paths, (str, bytes, bytearray)) or not isinstance(
            paths, Sequence
        ):
            raise TypeError(f"artifact tier {tier!r} must contain a path sequence")
        if not paths:
            raise ValueError(f"artifact tier {tier!r} must not be empty")
        resolved_paths = [
            (
                Path(path) if Path(path).is_absolute() else artifact_root / Path(path)
            ).resolve()
            for path in paths
        ]
        if len(resolved_paths) != len(set(resolved_paths)):
            raise ValueError(f"duplicate artifact in tier {tier!r}")
        for path in paths:
            relative = _portable_relative(Path(path), artifact_root)
            collision = _collision_key(relative)
            previous = collision_paths.setdefault(collision, relative)
            if previous != relative:
                raise ValueError(
                    "artifact paths collide under portable case normalization: "
                    f"{previous!r}, {relative!r}"
                )
            memberships.setdefault(relative, set()).add(tier)
    if not memberships:
        raise ValueError("artifact manifest must contain at least one file")

    records: list[dict[str, Any]] = []
    for relative, member_tiers in sorted(memberships.items()):
        path = artifact_root.joinpath(*_portable_path_parts(relative))
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
    aggregate_payload = {"producing_run": producing_run, "files": records}
    return {
        "schema_version": 1,
        "producing_run": producing_run,
        "files": records,
        "aggregate_sha256": sha256_json(aggregate_payload),
    }


def verify_artifact_manifest(
    root: str | Path,
    manifest: Mapping[str, Any],
) -> None:
    """Fail closed when a portable artifact manifest or any byte changes."""

    artifact_root = Path(root)
    if not artifact_root.is_dir():
        raise ValueError("artifact root must be an existing directory")
    artifact_root = artifact_root.resolve()
    if not isinstance(manifest, Mapping) or set(manifest) != {
        "schema_version",
        "producing_run",
        "files",
        "aggregate_sha256",
    }:
        raise ValueError("artifact manifest schema fields are invalid")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
    ):
        raise ValueError("artifact manifest schema_version mismatch")
    producing_run = _require_producing_run(manifest.get("producing_run"))
    records = manifest.get("files")
    if not isinstance(records, list):
        raise TypeError("artifact manifest files must be a list")
    if not records:
        raise ValueError("artifact manifest must contain at least one file")
    expected_aggregate = manifest.get("aggregate_sha256")
    _require_digest(expected_aggregate, field="aggregate_sha256", pattern=_SHA256)
    if (
        sha256_json({"producing_run": producing_run, "files": records})
        != expected_aggregate
    ):
        raise ValueError("artifact manifest aggregate hash mismatch")

    seen: set[str] = set()
    collision_paths: dict[str, str] = {}
    ordered_paths: list[str] = []
    validated_records: list[tuple[str, tuple[str, ...], int, str]] = []
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
        parts = _portable_path_parts(relative)
        if relative in seen:
            raise ValueError(f"duplicate artifact path: {relative}")
        seen.add(relative)
        ordered_paths.append(relative)
        collision = _collision_key(relative)
        previous = collision_paths.setdefault(collision, relative)
        if previous != relative:
            raise ValueError(
                "artifact paths collide under portable case normalization: "
                f"{previous!r}, {relative!r}"
            )
        size = record["bytes"]
        if type(size) is not int or size <= 0:
            raise ValueError(f"artifact size is invalid: {relative}")
        _require_digest(
            record["sha256"], field=f"sha256 for {relative}", pattern=_SHA256
        )
        member_tiers = record["tiers"]
        if (
            not isinstance(member_tiers, list)
            or not member_tiers
            or member_tiers != sorted(set(member_tiers))
            or any(
                not isinstance(tier, str) or _SAFE_TOKEN.fullmatch(tier) is None
                for tier in member_tiers
            )
        ):
            raise ValueError(f"artifact tiers are invalid: {relative}")
        validated_records.append((relative, parts, size, record["sha256"]))
    if ordered_paths != sorted(ordered_paths):
        raise ValueError("artifact file records are not in canonical path order")

    for relative, parts, size, expected_sha256 in validated_records:
        path = artifact_root.joinpath(*parts)
        observed_relative = _portable_relative(path, artifact_root)
        if observed_relative != relative:
            raise ValueError(f"artifact path canonicalization mismatch: {relative}")
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"artifact size mismatch: {relative}")
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"artifact hash mismatch: {relative}")
