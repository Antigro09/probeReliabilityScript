from __future__ import annotations

import json
from pathlib import Path

from scripts.run_reviewer_caveat_extension import COMMANDS, build_parser, main

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_extension_cli_exposes_every_stage_and_windows_devices() -> None:
    assert COMMANDS == (
        "preflight",
        "robustness",
        "construct-panel",
        "analyze",
        "package-artifacts",
        "figures",
        "patch-paper",
        "all",
    )
    parsed = build_parser().parse_args(
        [
            "preflight",
            "--config",
            str(PROJECT_ROOT / "revision_caveat_extension_spec.yaml"),
            "--base-config",
            str(PROJECT_ROOT / "revision_experiment_spec.yaml"),
            "--device",
            "cuda",
            "--dry-run",
        ]
    )
    assert parsed.device == "cuda"


def test_extension_cli_accepts_an_explicit_reproduced_base_run() -> None:
    parsed = build_parser().parse_args(
        [
            "all",
            "--config",
            str(PROJECT_ROOT / "revision_caveat_extension_spec.yaml"),
            "--base-config",
            str(PROJECT_ROOT / "revision_experiment_spec.yaml"),
            "--base-run",
            str(PROJECT_ROOT / "results" / "fresh-base-run"),
            "--allow-reproduced-base",
        ]
    )

    assert parsed.base_run == PROJECT_ROOT / "results" / "fresh-base-run"
    assert parsed.allow_reproduced_base is True


def test_extension_cli_dry_run_reports_locked_counts_without_execution(
    capsys, monkeypatch
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not import or execute the experiment runner")

    monkeypatch.setattr(
        "src.reviewer_revision.extension_runner.execute", forbidden
    )
    code = main(
        [
            "all",
            "--config",
            str(PROJECT_ROOT / "revision_caveat_extension_spec.yaml"),
            "--base-config",
            str(PROJECT_ROOT / "revision_experiment_spec.yaml"),
            "--dry-run",
            "--device",
            "cpu",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "dry_run_ok"
    assert payload["planned"]["confirmatory_cells"] == 11
    assert payload["planned"]["total_worker_edits"] == 715
    assert payload["planned"]["inferential_candidate_edits"] == 660
    assert payload["planned"]["compatibility_rows"] == 4290
    assert payload["planned"]["computes_confirmatory_endpoints"] is False
