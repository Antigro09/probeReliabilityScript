from __future__ import annotations

import copy

import pytest

from src.reviewer_revision.portability import (
    build_artifact_manifest,
    build_environment_lock,
    verify_artifact_manifest,
)


def test_environment_lock_is_canonical_and_machine_neutral(monkeypatch):
    monkeypatch.setattr(
        "src.reviewer_revision.portability.installed_distributions",
        lambda: [
            {"name": "PyYAML", "version": "6.0.2", "direct_url": None},
            {"name": "scikit_learn", "version": "1.7.1", "direct_url": None},
        ],
    )

    record = build_environment_lock(
        spec_hash="a" * 64,
        git_commit="b" * 40,
        python_version="3.12.7",
        platform_record={"system": "Windows", "machine": "AMD64"},
    )

    assert record["distributions"] == [
        {"name": "pyyaml", "version": "6.0.2"},
        {"name": "scikit-learn", "version": "1.7.1"},
    ]
    assert record["requirements"] == ["pyyaml==6.0.2", "scikit-learn==1.7.1"]
    assert len(record["aggregate_sha256"]) == 64
    assert "C:\\" not in repr(record)


@pytest.mark.parametrize(
    "distribution",
    [
        {
            "name": "private",
            "version": "1.0",
            "direct_url": "file:///C:/Users/name/private",
        },
        {
            "name": "private",
            "version": "1.0",
            "direct_url": "https://user:secret@example.test/archive.whl",
        },
        {"name": "private", "version": "1.0", "editable": True},
    ],
)
def test_environment_lock_rejects_nonportable_distribution_sources(
    monkeypatch, distribution
):
    monkeypatch.setattr(
        "src.reviewer_revision.portability.installed_distributions",
        lambda: [distribution],
    )

    with pytest.raises(ValueError, match="direct URL|editable"):
        build_environment_lock(spec_hash="a" * 64, git_commit="b" * 40)


def test_environment_lock_rejects_duplicate_canonical_names(monkeypatch):
    monkeypatch.setattr(
        "src.reviewer_revision.portability.installed_distributions",
        lambda: [
            {"name": "scikit-learn", "version": "1.7.1"},
            {"name": "scikit_learn", "version": "1.7.1"},
        ],
    )

    with pytest.raises(ValueError, match="duplicate distribution"):
        build_environment_lock(spec_hash="a" * 64, git_commit="b" * 40)


def test_artifact_manifest_unions_tiers_and_verifies_bytes(tmp_path):
    rows = tmp_path / "analysis" / "rows.json"
    rows.parent.mkdir()
    rows.write_text("{}\n", encoding="utf-8")
    summary = tmp_path / "summary.json"
    summary.write_text('{"status":"ok"}\n', encoding="utf-8")

    manifest = build_artifact_manifest(
        tmp_path,
        {
            "minimal": [summary],
            "analysis": [rows, summary],
        },
    )

    assert [record["path"] for record in manifest["files"]] == [
        "analysis/rows.json",
        "summary.json",
    ]
    assert manifest["files"][1]["tiers"] == ["analysis", "minimal"]
    verify_artifact_manifest(tmp_path, manifest)

    rows.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="size mismatch|hash mismatch"):
        verify_artifact_manifest(tmp_path, manifest)


def test_artifact_manifest_rejects_escape_empty_and_duplicate_tier_entry(tmp_path):
    empty = tmp_path / "empty.json"
    empty.touch()
    outside = tmp_path.parent / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="empty artifact"):
            build_artifact_manifest(tmp_path, {"analysis": [empty]})
        with pytest.raises(ValueError, match="escapes artifact root"):
            build_artifact_manifest(tmp_path, {"analysis": [outside]})
        with pytest.raises(ValueError, match="duplicate artifact"):
            build_artifact_manifest(tmp_path, {"analysis": [outside, outside]})
    finally:
        outside.unlink(missing_ok=True)


def test_artifact_manifest_rejects_tampered_structure_and_path_escape(tmp_path):
    artifact = tmp_path / "rows.json"
    artifact.write_text("{}\n", encoding="utf-8")
    manifest = build_artifact_manifest(tmp_path, {"analysis": [artifact]})

    altered = copy.deepcopy(manifest)
    altered["files"][0]["path"] = "../rows.json"
    with pytest.raises(ValueError, match="portable relative path|aggregate hash"):
        verify_artifact_manifest(tmp_path, altered)

    altered = copy.deepcopy(manifest)
    altered["files"].append(copy.deepcopy(altered["files"][0]))
    # Re-hashing a malformed list must not make duplicate paths acceptable.
    from src.reviewer_revision.artifacts import sha256_json

    altered["aggregate_sha256"] = sha256_json(altered["files"])
    with pytest.raises(ValueError, match="duplicate artifact path"):
        verify_artifact_manifest(tmp_path, altered)
