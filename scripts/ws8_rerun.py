"""
WS8 — v2 re-run orchestrator.

Sequences the repaired benchmark (scripts/run_benchmark_v2.py) over the 6 models
x 3 tasks. Each (model, task) runs as a SUBPROCESS so GPU memory is fully freed
between models (important on a 12 GB card with a 3B model in the set). Every
sub-run is resume-safe (WS0.1), so the whole sweep can be interrupted and
restarted and it will pick up where it left off.

Models are ordered small -> big so the cheap ones finish (and validate the
pipeline end to end) before the expensive ones start. Gated models
(Llama-3.2-3B, Gemma-2-2b) require HF license acceptance on the account; if a
sub-run fails to load one, the orchestrator records it and continues.

Usage:
    python -m scripts.ws8_rerun                     # full sweep, k=20
    python -m scripts.ws8_rerun --k 2 --smoke       # quick end-to-end check
    python -m scripts.ws8_rerun --models pythia gpt2 bert --tasks sva sst2
    python -m scripts.ws8_rerun --dry-run
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Small -> big. 'gated' models need HF license acceptance on the account.
MODELS = [
    {"key": "pythia", "config": "configs/pythia.yaml", "gated": False},
    {"key": "gpt2", "config": "configs/gpt2.yaml", "gated": False},
    {"key": "bert", "config": "configs/bert.yaml", "gated": False},
    {"key": "qwen", "config": "configs/qwen.yaml", "gated": False},
    {"key": "gemma", "config": "configs/gemma.yaml", "gated": True},
    {"key": "llama", "config": "configs/llama.yaml", "gated": True},
]
TASKS = ["sva", "gender", "sst2"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="*", default=[m["key"] for m in MODELS])
    p.add_argument("--tasks", nargs="*", default=TASKS)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--evaluators", type=int, default=5)
    p.add_argument("--smoke", action="store_true",
                   help="tiny end-to-end check: --k small, --no-tvd, capped examples")
    p.add_argument("--skip-gated", action="store_true")
    p.add_argument("--no-tvd", action="store_true")
    p.add_argument("--out-dir", default="results/benchmark_v2")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    by_key = {m["key"]: m for m in MODELS}
    selected = [by_key[k] for k in args.models if k in by_key]
    if args.skip_gated:
        selected = [m for m in selected if not m["gated"]]

    log_dir = PROJECT_ROOT / args.out_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / "ws8_sweep_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"runs": []}

    combos = [(m, t) for m in selected for t in args.tasks]
    print(f"WS8 sweep: {len(combos)} (model,task) runs  k={args.k}"
          f"{'  [SMOKE]' if args.smoke else ''}")

    for m, task in combos:
        cmd = [sys.executable, "-m", "scripts.run_benchmark_v2",
               "--config", m["config"], "--task", task,
               "--k", str(2 if args.smoke else args.k),
               "--evaluators", str(args.evaluators)]
        if args.no_tvd or args.smoke:
            cmd.append("--no-tvd")
        if args.smoke:
            cmd += ["--max-examples", "800"]
        print("\n" + "#" * 72)
        print(f"# {m['key']} x {task}   {'(gated)' if m['gated'] else ''}")
        print("#", " ".join(cmd))
        print("#" * 72)
        if args.dry_run:
            continue
        t0 = datetime.datetime.now(datetime.timezone.utc)
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        rec = {
            "model": m["key"], "task": task, "returncode": proc.returncode,
            "gated": m["gated"], "smoke": args.smoke,
            "started_utc": t0.isoformat(timespec="seconds"),
            "ended_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "status": ("ok" if proc.returncode == 0 else
                       "gate_failed" if proc.returncode == 2 else "error"),
        }
        summary["runs"].append(rec)
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"[{m['key']} x {task}] -> {rec['status']} (rc={proc.returncode})")

    print(f"\nSweep summary: {summary_path}")
    ok = sum(1 for r in summary["runs"] if r["status"] == "ok")
    print(f"  ok={ok}  total_recorded={len(summary['runs'])}")


if __name__ == "__main__":
    main()
