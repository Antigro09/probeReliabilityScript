"""Run every registered synthetic condition without overwriting outputs.

The earlier convenience command ran only the orthogonal conditions.  This
orchestrator adds the preregistered ``<v_s,v_c> in {0.0, 0.3}`` grid for both
linear and nonlinear generators, uses rank two for the nonlinear concept, and
stores each condition in its own directory.

Final run:
    python -m scripts.ws9_synthetic_controls

Quick pipeline check:
    python -m scripts.ws9_synthetic_controls --smoke
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALPHAS = "0.50,0.60,0.70,0.80,0.90,0.98"
CONDITIONS = [
    ("linear", 0.0, 1),
    ("linear", 0.3, 1),
    ("nonlinear", 0.0, 2),
    ("nonlinear", 0.3, 2),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replicates", type=int, default=20)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--control-draws", type=int, default=20)
    p.add_argument("--evaluators", type=int, default=5)
    p.add_argument("--D", type=int, default=256)
    p.add_argument("--N", type=int, default=8000)
    p.add_argument("--out-dir", default="results/robustness_v3/synthetic")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="passed through to scripts.ws5_run; see its --help for why "
                        "cpu can outrun cuda on this tiny synthetic workload")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-attempts", type=int, default=5,
                   help="a crashed condition is relaunched as a fresh subprocess up to "
                        "this many times; scripts.ws5_run resumes from whatever combos "
                        "already completed rather than starting over. Observed crashes "
                        "were native aborts uncorrelated with input data (two isolated "
                        "re-runs of crashed combos completed cleanly), consistent with "
                        "GPU/driver state degrading under many hours of continuous load "
                        "rather than a reproducible bug -- a fresh process reliably clears it.")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.out_dir) if Path(args.out_dir).is_absolute() else PROJECT_ROOT / args.out_dir
    root.mkdir(parents=True, exist_ok=True)
    summary = {"registered_alphas": [float(x) for x in ALPHAS.split(",")], "conditions": []}

    for family, corr, rank in CONDITIONS:
        name = f"{family}_corr_{corr:.1f}".replace(".", "p")
        condition_dir = root / name
        cmd = [
            sys.executable, "-m", "scripts.ws5_run",
            "--family", family,
            "--alphas", ("0.50,0.90" if args.smoke else ALPHAS),
            "--replicates", str(1 if args.smoke else args.replicates),
            "--seeds", str(1 if args.smoke else args.seeds),
            "--evaluators", str(args.evaluators),
            "--D", str(32 if args.smoke else args.D),
            "--N", str(800 if args.smoke else args.N),
            "--rank", str(rank),
            "--vs-vc-corr", str(corr),
            "--controls",
            "--control-draws", str(2 if args.smoke else args.control_draws),
            "--rlace-steps", str(20 if args.smoke else 200),
            "--out-dir", str(condition_dir),
            "--device", args.device,
        ]
        attempts = []
        returncode = 0
        if not args.dry_run:
            for attempt in range(1, args.max_attempts + 1):
                print("\n" + "#" * 78)
                print(f"# attempt {attempt}/{args.max_attempts}:", " ".join(cmd))
                print("#" * 78)
                returncode = subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode
                attempts.append(returncode)
                if returncode == 0:
                    break
                print(f"[{name}] attempt {attempt} failed with rc={returncode}; "
                      f"relaunching fresh (scripts.ws5_run resumes completed combos)")
        record = {
            "name": name,
            "family": family,
            "vs_vc_corr": corr,
            "rank": rank,
            "returncode": returncode,
            "attempts": attempts,
            "status": "ok" if returncode == 0 else "error",
            "out_dir": str(condition_dir),
        }
        head_path = condition_dir / f"headtohead_{family}.json"
        battery_path = condition_dir / f"battery_{family}.json"
        if head_path.exists():
            record["head_to_head"] = json.loads(head_path.read_text())
        if battery_path.exists():
            record["battery"] = json.loads(battery_path.read_text())
        summary["conditions"].append(record)
        if returncode != 0:
            print(f"[{name}] failed after {len(attempts)} attempt(s) (rcs={attempts}); "
                  f"continuing to preserve other conditions")

    if not args.dry_run:
        (root / "synthetic_condition_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True)
        )
    print(f"\nSynthetic condition summary: {root / 'synthetic_condition_summary.json'}")


if __name__ == "__main__":
    main()
