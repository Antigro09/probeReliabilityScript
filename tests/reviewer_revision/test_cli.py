from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path("scripts/run_reviewer_revision.py")
CONFIG = Path("revision_experiment_spec.yaml")
SUBCOMMANDS = (
    "preflight",
    "benchmark",
    "reproduce-baseline",
    "matched-split",
    "epsilon-sweep",
    "construct-check",
    "analyze",
    "figures",
    "patch-paper",
    "all",
)


@pytest.mark.parametrize("subcommand", SUBCOMMANDS)
def test_every_subcommand_supports_locked_common_options(subcommand, tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            subcommand,
            "--config",
            str(CONFIG),
            "--output-root",
            str(tmp_path),
            "--device",
            "cpu",
            "--log-level",
            "DEBUG",
            "--resume",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert '"command"' in completed.stdout
    assert subcommand in completed.stdout
    assert '"config_hash"' in completed.stdout


def test_invalid_config_returns_nonzero(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "preflight",
            "--config",
            str(tmp_path / "missing.yaml"),
            "--output-root",
            str(tmp_path),
            "--device",
            "cpu",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "missing.yaml" in completed.stderr


def test_device_choices_reject_cuda():
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "preflight",
            "--config",
            str(CONFIG),
            "--device",
            "cuda",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
