#!/usr/bin/env python
"""Run the locked August 2026 reviewer-revision pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reviewer_revision.config import load_revision_config

LOGGER = logging.getLogger("reviewer_revision.cli")

COMMANDS = (
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True, type=Path)
        subparser.add_argument(
            "--output-root",
            type=Path,
            default=Path("results/reviewer_revision_2026_08"),
        )
        subparser.add_argument("--resume", action="store_true")
        subparser.add_argument("--dry-run", action="store_true")
        subparser.add_argument(
            "--device", choices=("auto", "mps", "cpu"), default="auto"
        )
        subparser.add_argument(
            "--log-level",
            choices=("DEBUG", "INFO", "WARNING", "ERROR"),
            default="INFO",
        )
    return parser


def _dry_run_payload(command: str, config, args: argparse.Namespace) -> dict:
    return {
        "command": command,
        "config": str(args.config),
        "config_hash": config.config_hash,
        "output_root": str(args.output_root),
        "resume": bool(args.resume),
        "device": args.device,
        "planned": {
            "matched_split_full_cells": len(config.matched_split_cells("full")),
            "matched_split_fallback_cells": len(config.matched_split_cells("fallback")),
            "matched_split_full_rows": len(config.matched_split_row_keys("full")),
            "matched_split_fallback_rows": len(config.matched_split_row_keys("fallback")),
            "epsilon_required_rows": len(config.epsilon_sweep_row_keys()),
            "construct_edits": len(config.construct_edit_keys()),
        },
        "status": "dry_run_ok",
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        config = load_revision_config(args.config)
        if args.dry_run:
            print(json.dumps(_dry_run_payload(args.command, config, args), indent=2))
            return 0
        from src.reviewer_revision.runner import execute

        return int(execute(args.command, config, args))
    except Exception as exc:
        LOGGER.exception("reviewer revision command failed")
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
