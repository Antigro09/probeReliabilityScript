# Pre-Registration v2 — Alignment as a Predictor of *Repaired* Probe Reliability

**Locked:** 2026-07-06 (git tag `prereg-v2`)
**Status:** All thresholds, floors, margins, and data/design choices below are
fixed *before* the v2 benchmark and the synthetic study are run. Changing any of
them after seeing v2 results violates this pre-registration.
**Supersedes:** `prereg-v1` for all v2 (schema_version == 2) records. The v1
document and `scripts/predictor_eval.py` remain the sole authority for the v1
records and are untouched.

---

## 0. Why v2 exists (what v1 got wrong)

Three defects, established by reading the code and the v1 records (see
`docs/WS5_DESIGN.md` for the full account):

1. **The v1 target `R` does not depend on the probe it ranks.** In
   `run_benchmark.run_one_cell`, the validation probes and the five
   interventions are computed **once per (model, layer, task) cell** and never
   see the candidate probe, so `R` is identical across all 3 architectures × 20
   seeds. The v1 "rank-1 hit rate" was a forced 3-way tie; the v1 Spearman
   correlated a per-probe `A` against a target with no per-probe variance.
2. **`max`-over-methods + no decodability floor** let a degenerate method pin
   `R = 1.0` (Gemma: `R = 1.0` in all 900 records; the Ze denominator sat at
   chance). WS3 added the floor; `max` is dropped here.
3. **MKA was not a real architecture** (byte-identical to MLP in 1800/1800
   seed-matched records — a zero-gradient regularizer). WS1 fixed it.

v2 replaces the target with the **repaired reliability `R_rep`** (WS5), makes
MKA differentiable (WS1), rebuilds the gender task (WS2), floors the metrics
(WS3), and adds honest inference (WS6) and a TVD variant (WS7). The central
scientific question is unchanged: **does the cheap static alignment score `A`
predict a probe's causal reliability?** — but the target is now a metric that
(a) actually depends on the probe and (b) is validated against known ground
truth before it is trusted.

---

## 1. Locked definitions

### 1.1 Predictor `A` (unchanged from v1)
For a trained probe with flattened parameters `w` and the top-20 eigenvectors
`v_i` of its training-loss Hessian,
```
A = mean_{i=1..20} | cos(w, v_i) |          (src/hessian.py; top_n = 20)
```
Computed with no interventions. `A` never enters any reliability metric below
(enforced by a unit test, WS5 C-b).

### 1.2 Repaired reliability `R_rep` and the decomposition ladder (WS5)
For each candidate probe, an edit direction is the top-`r` right-singular
vector(s) of its per-example logit-difference Jacobian over a fixed reference
set (`src/ws5_repaired.candidate_direction`; for a linear probe this is exactly
`normalize(w[1]−w[0])`). The edit is the population projection
`X_inter @ (I − QQᵀ)`, scored by **E independent, bagged evaluator probes**
trained on a disjoint clean fold, with the WS3 floor applied, aggregated by
**mean over evaluators — never `max`**. The ladder (`build_ladder`):

| Rung | Edit source | Varies per probe? | Role |
|------|-------------|-------------------|------|
| `R_v1_max` | `max` over the 5 v1 methods | **no** (per cell) | the v1 object; reproduces Defect 1 |
| `R_v1_mean` | mean over the 5 v1 methods | no | isolates the `max`→mean effect |
| `R_excl` | mean over {INLP, RLACE} | **no** (per cell) | clean layer-level anchor, F1/F2-free |
| **`R_cand`** | the candidate's own subspace erasure | **yes** (per probe) | **the per-probe target** |

**Crucial and locked:** only `R_cand` varies per probe (INLP/RLACE/AlterRep/
FGSM/PGD edits are candidate-independent, so `R_excl`, `R_v1_*` are per-cell
constants — the v1 defect, retained on purpose for the head-to-head). Therefore
the **per-probe** claim "A predicts reliability" is a claim about **`R_cand`**;
`R_excl` supports only a **cell-level** claim. In v2 records the schema field
`"R"` is set to `R_cand` (so the locked evaluator `prereg_v2_eval.py` reads the
per-probe target directly); `R_v1`, `R_excl`, `R_v1_mean` ride alongside.

### 1.3 TVD reliability (WS7, secondary)
Canby et al. (arXiv:2408.15510) Eqs 1–4, computed inline from the oracle
probes' output distributions (`src/tvd.py`): nullifying completeness
`C = 1 − (k/(k−1))·δ(P̂,U)`, counterfactual `C = 1 − δ(P̂, onehot(z'))`,
selectivity `S = 1 − (1/m)·δ(P̂,P)`, reliability the harmonic mean. Reported
per method alongside the accuracy-based metrics; **not** the primary target.
Interpretation choices (a)/(b)/(c) in `src/tvd.py` were verified against Eqs
1–4; default aggregation `per_example` (the `dist` marginal cancels on balanced
data). k>2 counterfactual raises (binary v2 scope).

---

## 2. Locked benchmark setup

- **Models (6):** `EleutherAI/pythia-160m`, `openai-community/gpt2`,
  `google-bert/bert-base-uncased`, `Qwen/Qwen2.5-1.5B`, `google/gemma-2-2b`,
  `meta-llama/Llama-3.2-3B`. All 6 reported; none excluded.
- **Tasks (3):** SVA (Linzen 2016), Gender (Winogender full-sentence, WS2),
  SST-2. All 3 reported.
- **Architectures (3):** Linear, MLP, MKA (differentiable regularizer, WS1).
- **Layers (5):** `linspace(1, n_layers, 5)` (`select_layers`). All 5 reported.
- **Probe seeds:** K = 20 per (model, layer, task, arch) cell.
- **Extraction:** `hidden_states[layer]` at the last non-padding input token,
  `add_special_tokens=False`, single forward pass over all probed layers,
  cached with provenance (WS0.4). Representations are **z-scored per feature**
  using candidate-fold statistics before probe training (candidates,
  evaluators, and interventions all operate in this standardized space) — an
  invertible linear transform that keeps fixed-epoch probes from undertraining
  on large-scale residual streams; direction extraction and INLP-style erasure
  are unaffected.
- **4-way disjoint data split (WS5 §7.4), locked fractions (D11):**
  `{candidate: 0.40, evaluator: 0.20, intervention: 0.30, test: 0.10}`.
  The candidate trains on the candidate fold; evaluators (bagged) on the
  evaluator fold; the edit is applied and C/S measured on the intervention
  fold; candidate accuracy on the test fold. This is a v2 pipeline change;
  v1↔v2 candidate accuracies are **not** claimed comparable.
- **Learnability gate (WS2):** before launch, a linear probe at the middle
  probed layer must reach the registered floors and a shuffled-label control
  must sit at chance, for both Zc and Ze, or the run refuses (exit 2).
- **Output:** `results/benchmark_v2/<model>_<task>.jsonl`, schema_version = 2,
  stamped with git commit + data hash; resume-safe (WS0.1); analysis scripts
  take an explicit `--input-dir` and refuse mixed schema versions.

---

## 3. Locked thresholds and design choices

### 3.1 Metric floors (WS3)
- **Decodability floor** `DECODABILITY_FLOOR = 0.55` (must be > chance 0.5).
  Below it, C/S/R are **null**, never clipped; legacy clipped values and raw
  ratios ride alongside for diagnosis.

### 3.2 Learnability-gate floors (WS2)
- `zc_gate_floor = 0.60`, `ze_gate_floor = 0.55` (all tasks). Shuffled-label
  control band: chance ± max(0.05, 4·binomial-sd).

### 3.3 MKA regularization (WS1)
- `mka.lambda_reg = 0.5` (synthetic sweep: alignment rises monotonically with
  λ, parameters diverge from the seed-matched MLP from λ≥0.01, accuracy flat
  until a small dip at λ≥1; 0.5 is mid-range of the [0.3, 1.0] band where the
  regularizer visibly shapes training without cost). `knn_k = 10`. Hidden
  kernel: soft RBF, k-th-neighbor median bandwidth (detached); input kernel:
  hard binary kNN.

### 3.4 Gender task (WS2) — locked as regenerated (`data/gender.provenance.json`)
- Style **full-sentence** (pronoun included; last-token extraction tests gender
  persistence to sentence end). NEUTRAL occupations **dropped**. Skew from
  **BLS** %female (≥ 60 → FEM_SKEW, ≤ 40 → MASC_SKEW, else dropped).
  **Occupation** referent. 188 examples, balanced 4-cell, no both-labels rows.
  No synthetic fallback (deleted). *Note:* 188 is small; if the gender gate
  fails on a model, that model×gender cell is reported as a gate refusal, not
  silently run.

### 3.5 Repaired-metric knobs (WS5 §8, D1–D11)
- **D1** evaluators `E = 5`, `BAG_FRAC = 0.8`.
- **D2** report **both** `R_excl` (cell-level anchor) and `R_cand` (per-probe
  target); the per-probe claim is about `R_cand`, licensed non-circular by the
  synthetic dissociation (§4.2). `EVAL_SEED_BASE = 900000`.
- **D3** direction rank `r = 1`; reference-set size `N_dir = 4000`
  (capped at the intervention-fold size); `GRAD_NORM_MIN = 1e-6`.
- **D4** `FLOOR = 0.55` (= §3.1).
- **D5** synthetic shortcut grid `α_train ∈ {0.50,0.60,0.70,0.80,0.90,0.98}`,
  `α_eval = 0.5`.
- **D6** `μ_c = μ_e = μ_s = 2.0`, `σ = 1.0` (so the invariant concept is
  strongly decodable — Bayes acc ≈ 0.977 — and certifiers clear
  `EVAL_MIN_ACC = 0.90`; candidate confusion comes from the *shortcut* at high
  α, not a weak concept, resolving the D6↔D10 consistency requirement);
  non-orthogonal shortcut `⟨v_s,v_c⟩ = ρ ∈ {0.0, 0.3}`.
- **D7** candidate seeds `S = 8`, generator replicates `R = 20` (target; the
  pilot may justify more but not fewer for the powered head-to-head).
- **D8** battery: `ICC_MIN = 0.70`, `SAT_δ = 0.02`, `SAT_MAX = 0.20`,
  `RANGE_MIN = 0.30`, `SHUF_MAX = 0.10`, `RAND_MAX = 2·√(2/(πD))` (D-relative),
  `MARGIN_MIN = 0.30`, `SHORT_MAX = 0.10`, `A_HIGH = pilot P75(A)`.
- **D9** head-to-head: Kendall `τ_b`; replicate cluster-bootstrap CI, B = 2000;
  `TAU_MIN = 0.50`, `DELTA_MIN = 0.15`.
- **D10** `EVAL_MIN_ACC = 0.60` (out-of-bag), anchored to the learnability
  gate's zc floor. If a task is declared learnable at 0.60 on the candidate
  fold, a certifier on the eval fold must also reach ~0.60, else it is *blind*
  and the cell is invalid. Set here (not higher) because small models decode
  sentiment/gender at only ~0.65–0.75 at the last token; the WS3 per-metric
  floor (0.55) already nulls degenerate C/S, so this gate only rules out a
  broken certifier. A cell whose certifier cannot clear it is invalid, not
  scored low.
- **D11** 4-way split fractions = §2.

### 3.6 Honest-statistics knobs (WS6)
- `P1_RHO = 0.5`, `P1_P = 0.01`, `P2_HITRATE = 0.50`, `DISTINCT_R_MIN = 0.50`,
  `BOOTSTRAP_B = 2000`, `PERMUTATION_N = 10000`.

### 3.7 Attacker/evaluator control (WS4)
- `M = 5` seeded probe pairs; subset = all 6 models × middle layer ×
  {SVA, SST-2}; direction methods {AlterRep, FGSM, PGD}; controls {INLP, RLACE};
  bootstrap = 2000; `CI_ALPHA = 0.05`; `floor = 0.55`.

---

## 4. Predictions (locked)

### 4.1 Synthetic construct-validity of `R_rep` (A is NOT used)
Run on `src/ws5_synthetic.py` via `scripts/ws5_run.py`.

- **SP1** (R_rep tracks ground truth): `τ_rep = Kendall τ_b(R_cand, ρ_rec)
  ≥ TAU_MIN = 0.50`, permutation p < 0.01.
- **SP2** (R_rep beats v1): `Δτ = τ_rep − τ(R_v1_max, ρ_rec) ≥ DELTA_MIN =
  0.15`, replicate-clustered bootstrap CI lower bound > 0, conditional on SP1.
- **SP3** (validity battery): checks (i)–(v) all PASS, including (v) the
  **confident-shortcut dissociation** — a populated high-`A`/low-`R_cand`
  cell — evaluated on the **held-out non-linear family** (thresholds
  calibrated on the linear family).

If SP1–SP3 fail, `R_rep` is not a validated metric and the real-data claim in
§4.2 is **not** made; we report the failure.

### 4.2 Real-data — `A` predicts `R_cand` (primary, operational)
Run on `results/benchmark_v2/` via `scripts/prereg_v2_eval.py`. Precondition:
the **distinct-R gate passes** on `R_cand` (per-probe variance exists; if it
aborts, `R_cand` failed to depend on the probe and the whole approach is
refuted — reported as `ABORTED_NO_TARGET_VARIANCE`).

- **RP1 (primary):** across the 270 (median-A, median-`R_cand`) cells, Spearman
  `ρ ≥ 0.5`, permutation p < 0.01.
- **RP2 (secondary):** tie-aware rank-1 architecture hit rate `≥ 0.50` (now
  meaningful — `R_cand` varies across archs).
- **RP3 (robustness):** RP1 (`ρ ≥ 0.4`) and RP2 (`≥ 0.40`) hold within each of
  the 3 tasks.

### 4.3 Cell-level anchor (secondary, descriptive)
`median-A` vs `R_excl` Spearman across cells — a clean, floored analog of the
v1 claim (alignment as a *layer*-quality signal). Reported for interest; not a
pre-registered pass/fail (R_excl has no per-probe variance, so it cannot carry
the operational claim).

---

## 5. Outcome rubric (committed before seeing results)

| SP1–3 | RP1 | RP2 | RP3 | Interpretation |
|-------|-----|-----|-----|----------------|
| pass | ✓ | ✓ | ✓ | **Strong positive.** `R_rep` is a validated, non-circular reliability metric, and `A` operationally predicts per-probe reliability. |
| pass | ✓ | ✓ | ✗ | Aggregate + rank-1 utility, no per-task generalization. Reported honestly. |
| pass | ✓ | ✗ | * | `A` correlates with `R_cand` but does not select the best architecture. |
| pass | ✗ | * | * | **Negative but informative:** `R_rep` is valid (synthetic passed) yet `A` does not predict it. The predictor fails against a *correct* target — the sharpest possible negative. |
| fail | — | — | — | `R_rep` not validated; no real-data claim. The synthetic study explains why, and the metric is revised (not the thresholds). |

**We commit to publishing regardless of which row applies.** A confirmed
prediction gives an operational probe-selection criterion; a falsified one, now
tested against a validated target, is a real result about probe geometry.

---

## 6. What we will NOT do (anti-p-hacking)

1. **No threshold tuning.** Every number in §3–§4 is locked here.
2. **No model / task / architecture / layer exclusion.** All reported.
3. **No re-seeding** the benchmark to chase a result.
4. **No swapping the predictor.** `A` = mean |cos| with top-20 eigenvectors.
   Other formulations are exploratory, labeled, and do not become `A`.
5. **No using `R_legacy`** (the clipped v1 field) as the target or the
   comparator — that would re-inject the constant-1.0 pathology.
6. **No calibrating the battery on the family it is then judged on** — battery
   thresholds are locked on the linear family and evaluated on the non-linear
   family.

## 7. Pre-specified exploratory analyses (labeled, not predictions)
- TVD-based `R` vs accuracy-based `R` agreement (WS7).
- Attacker/evaluator matched-vs-split completeness gap (WS4).
- Per-layer-depth, per-architecture, and bottom-20-eigenvector analyses.
- `lambda_max` alone as a baseline predictor (expected uninformative).

## 8. Binding
Committed at git tag `prereg-v2`. The locked computations live in
`scripts/prereg_v2_eval.py` (real-data RP1–RP3), `scripts/ws5_run.py`
(synthetic SP1–SP3 + battery), `scripts/paper_stats.py` (shared honest stats),
and `src/{ws5_repaired,ws5_synthetic,tvd}.py`. The versions of those files at
`prereg-v2` determine whether the predictions are met.
