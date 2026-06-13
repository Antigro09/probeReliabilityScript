# Paper Data Summary — Pre-Registered Probe-Reliability Benchmark

Compiled 2026-06-10 from `results/benchmark/*.jsonl` (5,400 probe records)
and `results/benchmark/PREREG_OUTCOME.json` (locked evaluation output of
`scripts/predictor_eval.py` at tag `prereg-v1`).

All tables below other than Section 1 are **exploratory** (computed by
`scripts/paper_stats.py`, written after results were collected). Section 1
is the pre-registered analysis. Section 5 documents three implementation
problems discovered during this write-up that materially affect
interpretation; the paper must disclose them.

---

## 1. Pre-registered outcome (binding): NEGATIVE

Per the rubric in `PREREGISTRATION.md` (locked 2026-05-01, tag `prereg-v1`):

| Prediction | Threshold | Observed | Met |
|---|---|---|---|
| P1: Spearman ρ(A, R), 270 cells | ρ ≥ 0.5, p < 0.01 | **ρ = −0.170**, p = 0.0051, n = 270 | **NO** |
| P2: rank-1 hit rate, 90 cells | ≥ 50% | 76.7% (69/90) | yes (but see §5.3) |
| P3 gender | ρ ≥ 0.4 / HR ≥ 40% | ρ = −0.213 (p = 0.044) / 96.7% | **NO** |
| P3 sst2 | ρ ≥ 0.4 / HR ≥ 40% | ρ = −0.207 (p = 0.050) / 70.0% | **NO** |
| P3 sva | ρ ≥ 0.4 / HR ≥ 40% | ρ = −0.009 (p = 0.933) / 63.3% | **NO** |

**Outcome category: NEGATIVE** (P1 not met). Per the locked rubric, the paper
reports: "geometric and curvature signals correlate observationally but do
not transfer to operational probe selection."

Note the aggregate correlation is not merely below threshold — it is
*significantly negative* (ρ = −0.17, p = 0.005): higher Hessian-eigenvector
alignment weakly predicts *lower* intervention reliability, the opposite
direction of the registered hypothesis.

## 2. Benchmark scale (for the experimental-setup section)

- 6 models: BERT-base-uncased, GPT-2 (124M), Pythia-160M, Qwen2.5-1.5B,
  Gemma-2-2B, LLaMA-3.2-3B
- 3 tasks: SVA (Linzen), gender agreement, SST-2
- 3 nominal probe architectures: Linear, MLP, MKA-regularized MLP
  (but see §5.1 — MKA was effectively identical to MLP)
- 5 layers per model (linspace over depth), K = 20 seeds per cell
- 5,400 probe trainings + intervention evaluations; 270 aggregated
  (model, layer, task, arch) cells; 90 (model, layer, task) ranking cells
- Predictor A = mean |cos(w, v_i)| over top-20 Hessian eigenvectors
  (PyHessian power iteration); reliability R = max over INLP, RLACE,
  AlterRep, FGSM, PGD of harmonic_mean(completeness, selectivity)
- Total probe-level wallclock (sum of per-record `wallclock_s`): ≈ 6.5 h

## 3. Exploratory breakdowns (label as exploratory in the paper)

### Per model (45 cells each)

| Model | ρ(A,R) | p | rank-1 hits |
|---|---|---|---|
| Pythia-160M | −0.356 | 0.016 | 12/15 |
| Qwen2.5-1.5B | −0.444 | 0.002 | 10/15 |
| BERT-base | −0.187 | 0.220 | 13/15 |
| Gemma-2-2B | n/a (R constant at 1.0) | — | 9/15 |
| LLaMA-3.2-3B | +0.275 | 0.067 | 10/15 |
| GPT-2 | +0.187 | 0.219 | 15/15 |

### Per architecture (90 cells each)

| Arch | ρ(A,R) | median A | median R | median acc |
|---|---|---|---|---|
| linear | −0.245 (p = 0.020) | 0.0049 | 0.9999 | 0.672 |
| mlp | −0.130 (p = 0.221) | 0.0027 | 0.9999 | 0.700 |
| mka | identical to mlp (§5.1) | | | |

### Per layer position (1 = shallowest of 5 sampled; 54 cells each)

ρ(A,R): pos1 −0.067, pos2 −0.109, pos3 −0.311 (p = 0.022), pos4 −0.163,
pos5 −0.259 (p = 0.059). Weakly more negative in mid/deep layers.

### Pre-specified exploratory baseline

λ_max alone vs R (270 cells): ρ = +0.194, p = 0.0013 — small but, notably,
*better-signed* than the registered predictor A.

### Winning intervention method (per-probe argmax for R)

AlterRep 78.9%, INLP 11.1%, FGSM 6.7%, RLACE 3.3%, PGD 0%.

## 4. Raw data inventory

- `results/benchmark/<model>_<task>.jsonl` — 18 files, 5,400 records.
  Record schema: `arch, seed, accuracy, A, lambda_max, R, R_method,
  per_method{INLP,RLACE,AlterRep,FGSM,PGD}{C,S,R}, num_params, model,
  task, layer, wallclock_s`.
- `results/benchmark/PREREG_OUTCOME.json` — locked evaluation output.
- `scripts/paper_stats.py` — reproduces every exploratory number above.
- Provenance: `PREREGISTRATION.md` + `scripts/predictor_eval.py` are
  unchanged since tag `prereg-v1` (commit 8827e41). Three files carry
  uncommitted post-prereg fixes that do not touch the locked computation:
  `scripts/run_benchmark.py` (task-specific default data paths),
  `src/tasks.py` (empty-data fallback), `src/extraction.py`
  (`add_special_tokens=False` in the validation pass for consistency with
  extraction). These should be committed and disclosed as post-registration
  infrastructure changes.

## 5. Validity problems found during write-up (must be disclosed)

### 5.1 The MKA architecture is identical to MLP — all 1,800 records

Every MKA record is byte-for-byte identical to the corresponding MLP
record (same A, R, accuracy for all 1,800 (model, layer, task, seed)
combinations). Cause: `knn_kernel()` in `src/probes.py` builds a binary
0/1 adjacency matrix entirely under `torch.no_grad()`, so the MKA
regularizer is piecewise-constant with zero gradient w.r.t. probe
parameters. The regularizer adds a constant to the loss and never alters
training; with seed-controlled identical initialization, MKA and MLP
trajectories coincide exactly. The benchmark therefore compared **2**
architectures, not 3.

### 5.2 The gender task is unlearnable as run

`data/gender.tsv` sentences are truncated before the pronoun
("The technician told the customer that" carries both MASC and FEM
labels on identical text). The class label is not present in the input,
so Bayes-optimal accuracy is 50%; observed probe accuracy is *below*
chance (median 0.382, range 0.279–0.603) because probes overfit
contradictory labels. With acc_zc_pre < 0.5, the completeness formula
C = clip((pre − post)/(pre − 0.5), 0, 1) has a negative denominator and
is ill-defined, which contributes to spurious near-ceiling R on this
task. All gender-task numbers (including its 96.7% hit rate) are
artifacts.

### 5.3 Reliability R is saturated, so P2 and the ρ targets are degenerate

48.9% of all 5,400 records have R = 1.0 exactly; median R = 0.9999;
no record has R < 0.5 (5th percentile 0.95). Taking the max over five
intervention methods (AlterRep wins 79% of the time with C = 1.0)
pushes R to ceiling nearly everywhere. Consequences:

- In **90/90** ranking cells, at least two architectures have exactly
  tied median R. `predictor_eval.py` breaks ties by list order
  (`max()` returns the first maximizer), so the "rank-1 hit" is
  determined by tie-breaking, not by reliability differences. The
  76.7% hit rate is unchanged when the duplicated MKA arch is dropped,
  and the true chance baseline for the effective 2-way comparison is
  50%, not the 33.3% stated in the pre-registration.
- A near-constant target leaves almost no rank variance for any
  predictor to explain; the negative P1 is therefore partly a statement
  about the saturated reliability metric, not only about alignment.

### Recommendation

Per the pre-registration, the binding outcome is NEGATIVE and is
reported as such — the rubric committed to publishing this. The three
issues above do not rescue the hypothesis (none of them plausibly hides
a ρ ≥ 0.5 signal), but they change what the negative result *means* and
must appear in the limitations section. If the team wants conclusions
about alignment-vs-reliability per se, the benchmark needs a re-run with
(a) corrected gender data containing the pronoun, (b) a differentiable
or correctly applied MKA regularizer, and (c) a reliability target that
does not saturate (e.g., per-method R or mean over methods instead of
max). A re-run must be labeled a *deviation* from the pre-registration
with the bugs documented — it cannot silently replace the registered
result (PREREGISTRATION.md §"What We Will Not Do", item 6).
