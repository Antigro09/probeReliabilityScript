# v2 Results — Alignment does NOT predict repaired probe reliability

Run: `scripts/ws8_rerun.py` at tag `prereg-v2`. Records: `results/benchmark_v2/`
(2820 probe records, 141 cells). Locked evaluator: `scripts/prereg_v2_eval.py`
→ `results/benchmark_v2/PREREG_V2_OUTCOME.json`.

## Headline: OUTCOME = NEGATIVE (the informative kind)

- **Distinct-R gate PASSES** (0.0% of cells degenerate). R_cand genuinely varies
  per probe — the v1 defect (R identical across archs/seeds in a cell) is fixed.
  So the target is legitimate, and the negative result is *about the predictor*,
  not an artifact of a broken metric.
- **P1 (primary) NOT MET:** Spearman ρ(A, R_cand) = **0.009**, permutation
  p = 0.92, cluster-bootstrap 95% CI [−0.20, 0.21], n = 141 cells.
- **P2 (secondary) NOT MET:** tie-aware rank-1 hit rate = 0.38 (< 0.50).

The synthetic study (`scripts/ws5_run.py`) separately established that R_cand is
a *valid* metric: it ranks probes by ground-truth causal recovery at Kendall
τ = 0.74, beating v1 max-R (τ = 0.18) with Δτ = 0.55, CI [0.33, 0.82]. So A
fails against a **correct** target — the sharpest possible negative, and exactly
the prereg rubric's "R_rep valid yet A does not predict it" row.

## The decomposition ladder (why this matters)

ρ(A, ·) over 141 cell medians:

| target | ρ | p | note |
|---|---|---|---|
| R_cand (repaired, per-probe, validated) | +0.009 | 0.92 | **no relationship** |
| R_excl (INLP/RLACE, per-cell) | −0.155 | 0.07 | ~null |
| R_v1_max (v1-style max-over-5, broken) | **−0.368** | **<0.001** | significant but **wrong sign** |
| accuracy | +0.093 | 0.27 | null |

And ρ(accuracy, R_cand) = **−0.303** (p < 0.001): *more accurate probes are less
reliable* by R_cand — consistent with high-fit probes leaning on shortcuts whose
directions are less causally clean.

Per-architecture medians (A and R_cand move in **opposite** directions):

| arch | median A | median R_cand | median acc |
|---|---|---|---|
| linear | 0.0068 | 0.402 | 0.789 |
| mlp | 0.0040 | 0.482 | 0.807 |
| mka | 0.0007 | 0.475 | 0.802 |

Higher-capacity probes are *more* reliable but *less* aligned — so the arch with
the highest A (linear) is the least reliable, which is why P2 fails.

## Interpretation

Hessian directional alignment `A = mean|cos(w, top-20 eigenvectors)|` does not
predict causal probe reliability. The apparent v1 signal was an artifact of a
reliability metric that did not depend on the probe (and, where a v1-style metric
does vary, A correlates with it *negatively*). A itself is small (median 0.0034,
near the random-alignment floor) and, while it distinguishes architectures, it
carries no signal about which probe is causally reliable.

The paper's honest contribution: *a repaired, probe-dependent, ground-truth-
validated reliability metric, and the finding that geometric/curvature alignment
does not transfer to operational probe selection under it.*

## Data caveats (both caught by the WS2 learnability gate)

- **gemma-2-2b: EXCLUDED.** zc decodes at chance (~0.50) at the middle layer for
  every task → degenerate extraction (a known Gemma-2 hidden-state issue; likely
  attention soft-capping / bf16 / BOS handling). The gate refused it rather than
  emitting v1-style garbage. Fixing Gemma extraction is a follow-up.
- **gender: UNDERPOWERED.** 188 examples → tiny balanced 4-way folds; only
  pythia (120) and gpt2 (120) cells survived the gate. The full-sentence,
  last-token design needs more data or larger models. gender ρ = −0.51 on 12
  cells (p = 0.09) is not interpretable.

Usable models: pythia, gpt2, bert, qwen, llama (5). Tasks with broad coverage:
SVA, SST-2.

## Reproduce

```
python -m scripts.prereg_v2_eval --input-dir results/benchmark_v2   # P1/P2, distinct-R gate
python -m scripts.ws5_run --family linear --replicates 20 --seeds 8 --controls  # synthetic SP1-3
```
