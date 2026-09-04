#!/usr/bin/env python
"""Run the prospectively registered reviewer caveat-hardening extension."""

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
from src.reviewer_revision.extension_config import load_extension_config
from src.reviewer_revision.extension_runner import (
    DEFAULT_OUTPUT_ROOT,
    STAGES,
    build_dry_run_plan,
    execute,
)

LOGGER = logging.getLogger("reviewer_revision.extension.cli")
COMMANDS = (*STAGES, "all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True, type=Path)
        subparser.add_argument("--base-config", required=True, type=Path)
        subparser.add_argument(
            "--base-run",
            type=Path,
            help=(
                "validated base-run directory inside the repository; defaults to "
                "the registered immutable run"
            ),
        )
        subparser.add_argument(
            "--allow-reproduced-base",
            action="store_true",
            help=(
                "accept a newly reproduced, fully validated base run instead of "
                "requiring the registered generating commit"
            ),
        )
        subparser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        subparser.add_argument("--resume", action="store_true")
        subparser.add_argument("--dry-run", action="store_true")
        subparser.add_argument(
            "--device",
            choices=("auto", "cuda", "mps", "cpu"),
            default="auto",
        )
        subparser.add_argument(
            "--log-level",
            choices=("DEBUG", "INFO", "WARNING", "ERROR"),
            default="INFO",
        )
    return parser


def _dry_run_payload(command: str, extension, base, args: argparse.Namespace) -> dict:
    return {
        "command": command,
        "config": str(args.config),
        "config_hash": extension.config_hash,
        "base_config": str(args.base_config),
        "base_config_hash": base.config_hash,
        "base_run": str(args.base_run) if args.base_run is not None else None,
        "allow_reproduced_base": bool(args.allow_reproduced_base),
        "output_root": str(args.output_root),
        "resume": bool(args.resume),
        "device": args.device,
        "planned": build_dry_run_plan(extension, base),
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
        extension = load_extension_config(args.config)
        base = load_revision_config(args.base_config)
        if args.dry_run:
            print(json.dumps(_dry_run_payload(args.command, extension, base, args), indent=2))
            return 0
        return int(execute(args.command, extension, base, args))
    except Exception as exc:
        LOGGER.exception("reviewer caveat extension command failed")
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
