"""Resume-safe orchestrator for the schema-v3 robustness benchmark.

Each model-task combination runs in its own subprocess so GPU memory is fully
released.  The underlying runner resumes at (layer, architecture, seed)
granularity and records exclusions explicitly.  After the sweep, the analysis
script can build the paper tables and figures from whatever cells completed.

Final study:
    python -m scripts.ws9_robustness_sweep --save-predictions --analyze

Development smoke test:
    python -m scripts.ws9_robustness_sweep \
        --models pythia --tasks sva --smoke --analyze
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="*", default=[m["key"] for m in MODELS])
    p.add_argument("--tasks", nargs="*", default=TASKS)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--deep-k", type=int, default=5)
    p.add_argument("--evaluators", type=int, default=5)
    p.add_argument("--decoder-seeds", type=int, default=1)
    p.add_argument("--ranks", default="1,2,5")
    p.add_argument("--hessian-ks", default="1,5,10,20")
    p.add_argument("--residual-top-n", type=int, default=5)
    p.add_argument("--direction-resamples", type=int, default=5)
    p.add_argument("--group-mode", default="auto")
    p.add_argument("--gender-pre-pronoun", action="store_true",
                   help="also run the explicit surface-label leakage control")
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--skip-gated", action="store_true")
    p.add_argument("--skip-rlace", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _run(cmd, dry_run=False):
    print("\n" + "#" * 78)
    print("#", " ".join(cmd))
    print("#" * 78)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def main():
    args = parse_args()
    by_key = {m["key"]: m for m in MODELS}
    unknown = [key for key in args.models if key not in by_key]
    if unknown:
        raise SystemExit(f"unknown models: {unknown}; choose from {sorted(by_key)}")
    unknown_tasks = [task for task in args.tasks if task not in TASKS]
    if unknown_tasks:
        raise SystemExit(f"unknown tasks: {unknown_tasks}; choose from {TASKS}")
    selected = [by_key[key] for key in args.models]
    if args.skip_gated:
        selected = [model for model in selected if not model["gated"]]

    out_dir = args.out_dir or (
        "results/robustness_v3_smoke" if args.smoke else "results/robustness_v3"
    )
    cache_dir = args.cache_dir or (
        "cache/robustness_v3_smoke" if args.smoke else "cache/robustness_v3"
    )
    out_path = Path(out_dir) if Path(out_dir).is_absolute() else PROJECT_ROOT / out_dir
    out_path.mkdir(parents=True, exist_ok=True)
    summary_path = out_path / "ws9_sweep_summary.json"
    try:
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"runs": []}
    except json.JSONDecodeError:
        raise SystemExit(f"unreadable sweep summary: {summary_path}")

    combos = [(model, task, "last") for model in selected for task in args.tasks]
    if args.gender_pre_pronoun and "gender" in args.tasks:
        combos.extend((model, "gender", "pre-pronoun") for model in selected)

    for model, task, position in combos:
        cmd = [
            sys.executable, "-m", "scripts.run_robustness_v3",
            "--config", model["config"], "--task", task,
            "--position", position,
            "--k", str(1 if args.smoke else args.k),
            "--deep-k", str(1 if args.smoke else args.deep_k),
            "--evaluators", str(args.evaluators),
            "--decoder-seeds", str(1 if args.smoke else args.decoder_seeds),
            "--ranks", ("1" if args.smoke else args.ranks),
            "--hessian-ks", ("1,5" if args.smoke else args.hessian_ks),
            "--hessian-null-draws", str(500 if args.smoke else 10_000),
            "--residual-top-n", str(1 if args.smoke else args.residual_top_n),
            "--direction-resamples", str(2 if args.smoke else args.direction_resamples),
            "--group-mode", args.group_mode,
            "--out-dir", out_dir,
            "--cache-dir", cache_dir,
        ]
        if args.smoke:
            cmd += ["--max-examples", "800", "--perturbation-grid", "0.1,0.5", "--skip-rlace"]
        elif args.skip_rlace:
            cmd.append("--skip-rlace")
        if args.save_predictions:
            cmd.append("--save-predictions")

        started = datetime.datetime.now(datetime.timezone.utc)
        returncode = _run(cmd, dry_run=args.dry_run)
        ended = datetime.datetime.now(datetime.timezone.utc)
        status = "ok" if returncode == 0 else "excluded_by_gate" if returncode == 2 else "error"
        summary["runs"].append({
            "model": model["key"],
            "task": task,
            "position": position,
            "gated_model": model["gated"],
            "smoke": args.smoke,
            "returncode": returncode,
            "status": status,
            "started_utc": started.isoformat(timespec="seconds"),
            "ended_utc": ended.isoformat(timespec="seconds"),
        })
        if not args.dry_run:
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"[{model['key']} x {task} x {position}] -> {status} (rc={returncode})")

    if args.analyze:
        analysis_cmd = [
            sys.executable, "-m", "scripts.ws9_analyze",
            "--input-dir", out_dir,
        ]
        analysis_rc = _run(analysis_cmd, dry_run=args.dry_run)
        if analysis_rc != 0:
            raise SystemExit(analysis_rc)

    ok = sum(r["status"] == "ok" for r in summary["runs"])
    excluded = sum(r["status"] == "excluded_by_gate" for r in summary["runs"])
    print(f"\nSweep summary: {summary_path}")
    print(f"  ok={ok}  excluded_by_gate={excluded}  total_recorded={len(summary['runs'])}")


if __name__ == "__main__":
    main()
