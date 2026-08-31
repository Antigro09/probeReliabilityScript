from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from src.reviewer_revision.artifacts import sha256_json
from src.reviewer_revision.portability import (
    build_artifact_manifest,
    build_environment_lock,
    installed_distributions,
    verify_artifact_manifest,
)


def _requirements(tmp_path):
    path = tmp_path / "requirements.txt"
    path.write_text("torch>=2.5\npyyaml>=6\n", encoding="utf-8")
    return path


def _environment_kwargs(tmp_path):
    return {
        "spec_hash": "a" * 64,
        "git_commit": "b" * 40,
        "producing_run": "20260831T000000Z-bbbbbbb",
        "device_protocol": {
            "candidate_mlp_and_jacobian": "cpu",
            "statistics": "cpu",
        },
        "deterministic_algorithms": True,
        "requirements_path": _requirements(tmp_path),
        "python_version": "3.12.7",
        "python_abi": "cp312-win_amd64",
        "platform_record": {"system": "Windows", "machine": "AMD64"},
    }


def _standard_distributions():
    return [
        {
            "name": "PyYAML",
            "version": "6.0.2",
            "has_direct_url": False,
            "editable": False,
        },
        {
            "name": "scikit_learn",
            "version": "1.7.1",
            "has_direct_url": False,
            "editable": False,
        },
        {
            "name": "torch",
            "version": "2.5.1",
            "has_direct_url": False,
            "editable": False,
        },
    ]


def test_environment_lock_is_canonical_complete_and_machine_neutral(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "src.reviewer_revision.portability.installed_distributions",
        _standard_distributions,
    )

    record = build_environment_lock(**_environment_kwargs(tmp_path))

    assert [item["name"] for item in record["distributions"]] == [
        "pyyaml",
        "scikit-learn",
        "torch",
    ]
    assert record["requirements"] == [
        "pyyaml==6.0.2",
        "scikit-learn==1.7.1",
        "torch==2.5.1",
    ]
    assert record["python"] == {
        "version": "3.12.7",
        "implementation": "cpython",
        "abi": "cp312-win_amd64",
    }
    assert record["source_requirements"]["path"] == "requirements.txt"
    assert len(record["source_requirements"]["sha256"]) == 64
    assert record["pytorch_build"]["build_kind"] == "standard_release"
    assert record["device_protocol"] == {
        "candidate_mlp_and_jacobian": "cpu",
        "statistics": "cpu",
    }
    assert len(record["aggregate_sha256"]) == 64
    assert record == build_environment_lock(**_environment_kwargs(tmp_path))
    assert "C:\\" not in repr(record)


def test_environment_lock_redacts_direct_sources_and_labels_nightly_torch(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "src.reviewer_revision.portability.installed_distributions",
        lambda: [
            {
                "name": "en_core_web_sm",
                "version": "3.8.0",
                "has_direct_url": True,
                "editable": False,
            },
            {
                "name": "torch",
                "version": "2.12.0.dev20260408+cu128",
                "has_direct_url": False,
                "editable": False,
            },
        ],
    )

    record = build_environment_lock(**_environment_kwargs(tmp_path))

    assert record["requirements"] == []
    assert {item["source_kind"] for item in record["nonstandard_distributions"]} == {
        "direct_url_redacted",
        "nonstandard_build",
    }
    assert record["pytorch_build"] == {
        "version": "2.12.0.dev20260408+cu128",
        "source_kind": "nonstandard_build",
        "reconstructible_requirement": False,
        "build_kind": "nonstandard_or_direct_source",
    }
    serialized = repr(record).lower()
    assert "http://" not in serialized
    assert "https://" not in serialized
    assert "users/" not in serialized


@pytest.mark.parametrize(
    "name,version",
    [
        ("evil\n--extra-index-url example.test", "1.0"),
        ("evil==other", "1.0"),
        ("evil;whoami", "1.0"),
        (" evil", "1.0"),
        ("evil", " 1.0 "),
    ],
)
def test_environment_lock_rejects_requirement_injection(
    monkeypatch, tmp_path, name, version
):
    monkeypatch.setattr(
        "src.reviewer_revision.portability.installed_distributions",
        lambda: [
            {"name": name, "version": version},
            {"name": "torch", "version": "2.5.1"},
        ],
    )

    with pytest.raises(ValueError, match="invalid name|canonicalizable|noncanonical"):
        build_environment_lock(**_environment_kwargs(tmp_path))


def test_environment_lock_rejects_editable_and_duplicate_distributions(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "src.reviewer_revision.portability.installed_distributions",
        lambda: [
            {"name": "private", "version": "1.0", "editable": True},
            {"name": "torch", "version": "2.5.1"},
        ],
    )
    with pytest.raises(ValueError, match="editable"):
        build_environment_lock(**_environment_kwargs(tmp_path))

    monkeypatch.setattr(
        "src.reviewer_revision.portability.installed_distributions",
        lambda: [
            {"name": "scikit-learn", "version": "1.7.1"},
            {"name": "scikit_learn", "version": "1.7.1"},
            {"name": "torch", "version": "2.5.1"},
        ],
    )
    with pytest.raises(ValueError, match="duplicate distribution"):
        build_environment_lock(**_environment_kwargs(tmp_path))


@pytest.mark.parametrize("length", [7, 39, 41, 63])
def test_environment_lock_rejects_abbreviated_git_object_ids(
    monkeypatch, tmp_path, length
):
    monkeypatch.setattr(
        "src.reviewer_revision.portability.installed_distributions",
        _standard_distributions,
    )
    kwargs = _environment_kwargs(tmp_path)
    kwargs["git_commit"] = "b" * length
    with pytest.raises(ValueError, match="git_commit"):
        build_environment_lock(**kwargs)


def test_environment_lock_does_not_default_explicit_invalid_values(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "src.reviewer_revision.portability.installed_distributions",
        _standard_distributions,
    )
    kwargs = _environment_kwargs(tmp_path)
    kwargs["python_version"] = ""
    with pytest.raises(ValueError, match="python_version"):
        build_environment_lock(**kwargs)
    kwargs = _environment_kwargs(tmp_path)
    kwargs["platform_record"] = {}
    with pytest.raises(ValueError, match="platform record"):
        build_environment_lock(**kwargs)


def test_installed_distribution_helper_never_returns_raw_direct_url(monkeypatch):
    fake = SimpleNamespace(
        metadata={"Name": "private"},
        name="private",
        version="1.0",
        read_text=lambda name: (
            '{"url":"https://user:secret@example.test/private.whl"}'
            if name == "direct_url.json"
            else None
        ),
    )
    monkeypatch.setattr(
        "src.reviewer_revision.portability.importlib.metadata.distributions",
        lambda: [fake],
    )

    assert installed_distributions() == [
        {
            "name": "private",
            "version": "1.0",
            "has_direct_url": True,
            "editable": False,
        }
    ]


def test_real_project_environment_can_be_sanitized_without_urls(tmp_path):
    kwargs = _environment_kwargs(tmp_path)
    record = build_environment_lock(**kwargs)

    assert record["distributions"]
    assert any(item["name"] == "torch" for item in record["distributions"])
    assert "direct_url" not in repr(record).lower().replace("direct_url_redacted", "")
    assert "https://" not in repr(record).lower()


def _manifest(tmp_path):
    rows = tmp_path / "analysis" / "rows.json"
    rows.parent.mkdir()
    rows.write_text("{}\n", encoding="utf-8")
    summary = tmp_path / "summary.json"
    summary.write_text('{"status":"ok"}\n', encoding="utf-8")
    return build_artifact_manifest(
        tmp_path,
        {
            "minimal": [summary],
            "analysis": ["analysis/rows.json", "summary.json"],
        },
        producing_run="run-123",
    )


def _rehash(manifest):
    manifest["aggregate_sha256"] = sha256_json(
        {
            "producing_run": manifest["producing_run"],
            "files": manifest["files"],
        }
    )


def test_artifact_manifest_unions_tiers_records_run_and_verifies_bytes(tmp_path):
    manifest = _manifest(tmp_path)

    assert manifest["producing_run"] == "run-123"
    assert [record["path"] for record in manifest["files"]] == [
        "analysis/rows.json",
        "summary.json",
    ]
    assert manifest["files"][1]["tiers"] == ["analysis", "minimal"]
    verify_artifact_manifest(tmp_path, manifest)

    (tmp_path / "analysis" / "rows.json").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="size mismatch|hash mismatch"):
        verify_artifact_manifest(tmp_path, manifest)


def test_artifact_manifest_builds_and_verifies_with_relative_root(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "run-dir"
    artifact = run_dir / "nested" / "rows.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    manifest = build_artifact_manifest(
        "run-dir",
        {"analysis": ["nested/rows.json"]},
        producing_run="run-123",
    )

    verify_artifact_manifest("run-dir", manifest)


def test_artifact_manifest_rejects_escape_empty_and_duplicate_tier_entry(tmp_path):
    empty = tmp_path / "empty.json"
    empty.touch()
    outside = tmp_path.parent / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="empty artifact"):
            build_artifact_manifest(
                tmp_path, {"analysis": [empty]}, producing_run="run-123"
            )
        with pytest.raises(ValueError, match="escapes artifact root"):
            build_artifact_manifest(
                tmp_path, {"analysis": [outside]}, producing_run="run-123"
            )
        with pytest.raises(ValueError, match="duplicate artifact"):
            build_artifact_manifest(
                tmp_path,
                {"analysis": [outside, outside]},
                producing_run="run-123",
            )
        with pytest.raises(ValueError, match="must not be empty"):
            build_artifact_manifest(tmp_path, {"analysis": []}, producing_run="run-123")
    finally:
        outside.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "invalid_path",
    [
        "../rows.json",
        "C:/private/rows.json",
        "CON.txt",
        "bad?.json",
        "trailing./rows.json",
    ],
)
def test_artifact_verifier_rejects_nonportable_paths(tmp_path, invalid_path):
    manifest = _manifest(tmp_path)
    manifest["files"][0]["path"] = invalid_path
    _rehash(manifest)
    with pytest.raises(ValueError, match="portable|Windows|device name"):
        verify_artifact_manifest(tmp_path, manifest)


def test_artifact_verifier_rejects_duplicate_casefold_and_noncanonical_order(tmp_path):
    manifest = _manifest(tmp_path)
    duplicate = copy.deepcopy(manifest)
    duplicate["files"].append(copy.deepcopy(duplicate["files"][0]))
    _rehash(duplicate)
    with pytest.raises(ValueError, match="duplicate artifact path"):
        verify_artifact_manifest(tmp_path, duplicate)

    collision = copy.deepcopy(manifest)
    collision["files"][0]["path"] = "A.json"
    collision["files"][1]["path"] = "a.json"
    _rehash(collision)
    with pytest.raises(ValueError, match="collide"):
        verify_artifact_manifest(tmp_path, collision)

    unordered = copy.deepcopy(manifest)
    unordered["files"].reverse()
    _rehash(unordered)
    with pytest.raises(ValueError, match="canonical path order"):
        verify_artifact_manifest(tmp_path, unordered)


def test_artifact_verifier_rejects_empty_and_boolean_schema(tmp_path):
    manifest = _manifest(tmp_path)
    empty = copy.deepcopy(manifest)
    empty["files"] = []
    _rehash(empty)
    with pytest.raises(ValueError, match="at least one file"):
        verify_artifact_manifest(tmp_path, empty)

    boolean = copy.deepcopy(manifest)
    boolean["schema_version"] = True
    with pytest.raises(ValueError, match="schema_version"):
        verify_artifact_manifest(tmp_path, boolean)
