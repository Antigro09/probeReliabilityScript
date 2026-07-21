# WS4–WS7: what shipped, how to run it, what the PI must lock

Companion to `docs/WS5_DESIGN.md` (signed off) and `docs/WS0-WS3_DECISIONS_PENDING.md`.
Everything here is the code that does **not** depend on the locked prereg-v2
numbers (mechanisms only; thresholds are placeholders). WS8 (the real-data
re-run driver + wiring R_rep/TVD into `run_benchmark.py`, plus the 4-way data
split from WS5_DESIGN §7.4) is **gated on you locking `PREREGISTRATION_v2.md`**
and is not built yet.

## Files delivered

| WS | Files | Runs without models? |
|----|-------|----------------------|
| WS5 | `src/ws5_repaired.py`, `src/ws5_synthetic.py`, `scripts/ws5_run.py`, `tests/test_ws5_{repaired,synthetic}.py` | synthetic study yes (no HF models); needs torch |
| WS6 | `scripts/paper_stats.py` (additive), `scripts/prereg_v2_eval.py`, `tests/test_ws6_stats.py` | **yes — numpy/scipy only, 23 tests pass here** |
| WS7 | `src/tvd.py`, `tests/test_tvd.py` | kernels yes; `compute_tvd_metrics` needs torch |
| WS4 | `scripts/ws4_attacker_evaluator.py`, `tests/test_ws4.py` | core fn on synthetic reps yes; CLI needs cached reps |

## How to run (on the GPU box, torch present)

```
# WS6 honest stats (numpy/scipy only — already green here)
python -m pytest tests/test_ws6_stats.py -q
python -m scripts.prereg_v2_eval --input-dir results/benchmark_v2   # expect ABORT until R_rep lands (see below)

# WS5 synthetic construct-validity study (no HF models needed)
python -m scripts.ws5_run --family linear --replicates 5 --seeds 4 --controls
python -m scripts.ws5_run --family nonlinear --rank 2 --controls

# WS4 attacker/evaluator control (needs cached clean reps first)
python -m scripts.run_benchmark --config configs/<m>.yaml --task sva --cache-dir cache/benchmark_v2
python -m scripts.ws4_attacker_evaluator --cache-dir cache/benchmark_v2

# full suite (torch tests collect on the GPU box; here they skip/err on missing torch)
python -m pytest tests/ -q
```

`prereg_v2_eval.py` will emit `ABORTED_NO_TARGET_VARIANCE` on today's benchmark
records **by design**: v2 R is identical across archs×seeds in a cell (Defect 1),
so the distinct-R gate fires. That is the honest signal, not a tooling failure.
P1/P2 only become meaningful once a runner makes R depend on the evaluated
probe (WS5's R_rep, wired in at WS8).

## PI decisions to lock in `PREREGISTRATION_v2.md`

Beyond the WS5_DESIGN §8 table (D1–D11) and the WS0–WS3 items:

- **WS6** `prereg_v2_eval.py`: `P1_RHO=0.5`, `P1_P=0.01`, `P2_HITRATE=0.50`,
  `DISTINCT_R_MIN=0.50` (the gate that decides whether the eval aborts — ratify
  it explicitly), `BOOTSTRAP_B=2000`, `PERMUTATION_N=10000`.
- **WS7** `tvd.py`: `DEFAULT_TVD_MODE` and the three interpretation choices
  (a)/(b)/(c) — **verify against arXiv:2408.15510 Eqs 1–4 personally** (see
  residual risks). `DEFAULT_CHANCE=0.5`.
- **WS4** `ws4_attacker_evaluator.py`: `DEFAULT_M`, `DEFAULT_BOOTSTRAP`,
  `CI_ALPHA`, `DIRECTION_METHODS`, `CONTROL_METHODS`, model/task subset, floor.

## Residual risks the reviewers flagged (read before trusting green)

- **WS7 / TVD attribution (highest priority — you flagged this).** The default
  aggregation was changed from `dist` to `per_example` because `dist` reports
  completeness ≈ 1 for a *no-op* on balanced confident data (marginal
  cancellation — the WS3 pathology re-entering). The TVD formulas match Canby
  Eqs 1–4 as I read them, but **choices (a) per-example vs marginal, (b) floor
  reuse, (c) counterfactual target z'=1−zc, and the Eq3 selectivity oracle** are
  our reading. Confirm against the paper before the TVD column is trusted; the
  chosen `mode` is stamped into `TVDMetrics.as_dict()` for provenance.
- **WS6 Windows console.** `prereg_v2_eval.py` prints ✅/❌; on a cp1252 shell
  set `PYTHONIOENCODING=utf-8` (matches existing `predictor_eval.py` behavior).
  Also: the `paper_stats` WS6 bootstrap/permutation `cells` still carry the
  legacy clipped R (that script's existing convention); the authoritative
  floored-R analysis is `prereg_v2_eval.py`. Note it so the two rho values are
  not conflated in the paper.
- **WS4 soft test.** `test_matched_completeness_not_lower_than_split` is a
  one-sided `TOL=0.10` assertion; on an unlucky draw an evaluator with a sharper
  boundary could flip it. Run a few seeds before trusting green.
- **WS5 real-data wiring is a WS8 change, not a patch.** R_rep needs the 4-way
  disjoint split (candidate/evaluator/intervention/test); adopting it changes
  the benchmark data pipeline and is deliberately deferred to WS8 (post-lock).
```
