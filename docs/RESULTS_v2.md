# v2 Results — Alignment does NOT predict repaired probe reliability

Run: `scripts/ws8_rerun.py` at tag `prereg-v2` (+ the Gemma left-padding fix).
Records: `results/benchmark_v2/` (**3360 probe records, 168 cells, all 6 models**).
Locked evaluator: `scripts/prereg_v2_eval.py` →
`results/benchmark_v2/PREREG_V2_OUTCOME.json`. Figures: `results/figures/`.

## Headline: OUTCOME = NEGATIVE (the informative kind)

- **Distinct-R gate PASSES** (0.0% of cells degenerate). R_cand genuinely varies
  per probe — the v1 defect (R identical across archs/seeds in a cell) is fixed.
  So the target is legitimate, and the negative result is *about the predictor*,
  not an artifact of a broken metric.
- **P1 (primary) NOT MET:** Spearman ρ(A, R_cand) = **−0.042**, permutation
  p = 0.59, cluster-bootstrap 95% CI [−0.23, 0.15], n = 168 cells.
- **P2 (secondary) NOT MET:** tie-aware rank-1 hit rate = 0.375 (< 0.50).

(The 5-model result before the Gemma fix was ρ = +0.009; adding Gemma left it
firmly null — the finding is robust to the model set.)

The synthetic study (`scripts/ws5_run.py`) separately established that R_cand is
a *valid* metric: it ranks probes by ground-truth causal recovery at Kendall
τ = 0.74, beating v1 max-R (τ = 0.18) with Δτ = 0.55, CI [0.33, 0.82]. So A
fails against a **correct** target — the sharpest possible negative, and exactly
the prereg rubric's "R_rep valid yet A does not predict it" row.

## The decomposition ladder (why this matters)

ρ(A, ·) over 168 cell medians (6 models):

| target | ρ | p | note |
|---|---|---|---|
| R_cand (repaired, per-probe, validated) | −0.042 | 0.59 | **no relationship** |
| R_excl (INLP/RLACE, per-cell) | −0.149 | 0.05 | ~null |
| R_v1_max (v1-style max-over-5, broken) | **−0.322** | **<0.001** | significant but **wrong sign** |
| accuracy | +0.118 | 0.13 | null |

And ρ(accuracy, R_cand) = **−0.190** (p = 0.014): *more accurate probes are less
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

## Data notes

- **gemma-2-2b: RECOVERED.** Initially gate-refused (zc at chance ~0.50 on every
  task). Root cause was a real bug: `_last_token_hidden` used
  `attention_mask.sum()-1`, correct only for right-padding, but Gemma's tokenizer
  left-pads, so extraction landed mid-sequence and collapsed the reps. Fixed to
  compute the last real-token index for either padding side (a no-op for the 5
  right-padded models). Post-fix Gemma decodes SVA at 0.84–0.89 and contributes
  sva (240) + sst2 (300) cells. The WS2 gate correctly refused the garbage until
  the bug was fixed — it never entered the results silently.
- **gender: UNDERPOWERED.** 188 examples → tiny balanced 4-way folds; only pythia
  and gpt2 survived the gate (gemma/bert/qwen gender gate-failed; llama gender
  produced 0 usable cells). The full-sentence, last-token design needs more data
  or is dropped. Gender is not interpretable at this N.

All 6 models contribute (via SVA and/or SST-2); SVA and SST-2 have broad
coverage. The negative result holds across the full model set.

## Reproduce

```
python -m scripts.prereg_v2_eval --input-dir results/benchmark_v2   # P1/P2, distinct-R gate
python -m scripts.ws5_run --family linear --replicates 20 --seeds 8 --controls  # synthetic SP1-3
```
