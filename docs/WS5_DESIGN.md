# WS5 Design — Repaired Reliability Metric (R_rep) + Synthetic Validation

**Status:** DRAFT for PI sign-off · **Schema target:** v2 · **Owner:** WS5
**Depends on:** WS0.4 (rep cache), WS2 (learnability gate), WS3 (decodability floor).
**Blocks:** the paper's "alignment `A` predicts reliability" claim, and WS8's real-data schema (this doc adds a data-split change to the re-run — see §7.4).

> This document was written against a full read of `src/{probes,interventions,metrics,hessian,gates,tasks,extraction,repro}.py` and `scripts/run_benchmark.py`, then stress-tested by three independent adversarial reviews (circularity, statistics, feasibility). The design below is the **post-critique** version: several claims in the first internal draft were false and have been corrected here. Where a decision is the experimenter's, it is flagged for `PREREGISTRATION_v2.md` in §8, not decided here.

---

## 0. Executive summary

The paper's thesis is that a cheap static predictor
`A(probe) = mean_{i≤20} |cos(flat_params(probe), v_i)|` (top-20 Hessian eigenvectors of the probe's training loss; `hessian.py:108,138`, `mean_align_top` at `:77`) predicts a probe's **causal reliability** `R`. WS5 rebuilds `R` because the v1 `R` is broken in two independent ways, and the obvious fix introduces a third.

- **Defect 1 — v1 `R` does not depend on the probe it is meant to rank.** In `run_benchmark.py`, `val_probes` and `interventions_dict` are computed **once per layer** (`:462–474`) and `run_one_cell` never passes the candidate `probe` into interventions or metrics. So within any `(model, layer, task)` cell, `R`, `R_legacy`, and `per_method` are **identical across all 3 architectures × 20 seeds**; only `A`, `accuracy`, `lambda_max`, `num_params` vary per probe. The "rank-1 hit rate" (v1 prereg P2) is therefore a forced 3-way tie in every cell, and the Spearman claim (P1) correlates a per-probe `A` against a target with only 90 distinct values (one per layer/task/model), each repeated 3×. This is the deepest reason the v1 result is uninterpretable — deeper than the Gemma-constant-1.0 or MKA≡MLP defects, which sit on top of it.
- **Defect 2 — `max` over methods + no decodability floor** let a single degenerate method pin `R` at 1.0 (WS3 fixed the floor; `max` remains).
- **The trap — circularity (the "ydqQ" critique).** The natural fix ("derive the intervention from the candidate probe so `R` becomes probe-dependent") makes `R` a functional of the same trained weights `w` that `A` reads. For a linear probe the candidate direction is *exactly* `w[1]−w[0]` (identical to the AlterRep direction, `interventions.py:175`), so a correlation `A↔R` could be a mechanical identity with **no causal content**.

**Resolution (this design).** R_rep is the reliability of *the candidate probe's own extracted direction*, but (a) the direction is extracted architecture-neutrally and the **primary non-circularity evidence comes from a synthetic study where `A` is never computed** and ground-truth causal recovery is known; (b) the decisive test is **dissociation** — R_rep must score a *confident-but-wrong* probe LOW while its `A` is HIGH; (c) an **environment-shift** generator is what makes "causal" identifiable and the confident-shortcut probe constructible; (d) the edit is certified by **data-independent, bagged evaluator probes** on a disjoint fold; (e) methods are aggregated by **mean and reported as a decomposition ladder**, never `max`. If, and only if, R_rep is constructively dissociable from `A` on synthetic data, an empirical `A↔R_rep` correlation on real data is a genuine finding rather than an artifact.

The rest of this doc makes each of those buildable and decision-complete.

---

## 1. The circularity critique, stated precisely

### 1.1 Objects

- `A(P) = mean_i |cos(flat_params(P), v_i)|`, `v_i` = top-20 eigenvectors of `P`'s **training-loss** Hessian on `(X_cand, zc)` (`run_benchmark.py:226–231`, `hessian.py:120,177`). For `LinearProbe`, `flat_params = cat(w.flatten(), b)` with `w ∈ R^{2×D}` (`probes.py:44,60`).
- v1 `R` for a cell = `max_{m∈{INLP,RLACE,AlterRep,FGSM,PGD}} harmonic_mean(C_m,S_m)` with `C,S` from fixed clean-rep `LinearProbe` validators (`metrics.py:73`, `compute_intervention_metrics:241`).

### 1.2 Where each v1 method gets its edit direction

| Method | Edit direction source | Coupling to probe-under-test |
|---|---|---|
| INLP | fresh internal `LinearProbe`, row `w[0:1]` (`interventions.py:70–72`) | none (shares only `X,zc`) |
| RLACE | fresh internal adversarial subspace `Q` (`interventions.py:111,143`) | none (shares only `X,zc`) |
| AlterRep | `validation_probe.weight[1]−weight[0]` (`interventions.py:175`) | to the **validation** probe |
| FGSM | `sign(∇_X CE(validation_probe(X), zc))` (`interventions.py:206–209`) | to the **validation** probe |
| PGD | iterated FGSM gradient (`interventions.py:232–235`) | to the **validation** probe |

Crucially, `run_all_interventions` passes `val_probes.zc_probe` (a *validation* LinearProbe) as `validation_probe` (`run_benchmark.py:472`). So in v1 **no** method's direction comes from the arch-varying candidate — which is exactly Defect 1. WS5 must make the candidate matter *without* making `R` a trivial readout of `w`.

### 1.3 The three trivializing failure modes

- **F1 — direction identity.** If the edit direction is a function of the candidate's `w` and `A` is a function of `w`, `A↔R` can arise from shared dependence on `w`'s **direction** (not merely its magnitude). Unit-normalizing the edit direction removes only the magnitude channel; the direction channel — the one that matters — is untouched. For the linear arch the edit direction *is* `normalize(w[1]−w[0])`, so F1 is satisfied *by construction* unless mitigated. **This is the load-bearing risk and the first draft's "A ⊥ d_c-direction" claim was false.**
- **F2 — evaluator = actor.** If the probe that defines the edit also certifies that it "worked," a confident probe carves out a linear shortcut it can then destroy, inflating completeness regardless of causal reality.
- **F3 — `max` opportunism.** `max` over methods lets the single most self-aligned method set `R`, degenerating `R` toward "how confidently does this probe separate."

### 1.4 Decision-complete definition of "trivial"

The `A↔R_rep` link is declared **non-trivial** iff **all** hold:
1. **(Dissociability)** On the synthetic generator there exist candidate probes with `A ≥ A_HIGH` and `R_rep ≤ SHORT_MAX` (confident-shortcut) *and* probes with defensible `A` and high `R_rep` (genuine) — i.e. the {high-A}×{low/high-R_rep} quadrants are populated. `A` is **not computed** when defining R_rep or ground truth.
2. **(Ground-truth tracking, A-free)** R_rep ranks candidates by their recovery of the **planted invariant causal axis** `v_c` (Kendall `τ_rep ≥ TAU_MIN`), where recovery is measured against known `v_c`, `A` absent.
3. **(Independence in force)** Deliberately breaking evaluator independence (§4.2 C-c) measurably inflates R_rep and its `A`-correlation — proving independence is doing work.

Sections 2–6 build exactly this. §8 lists every threshold as a `PREREGISTRATION_v2.md` knob.

---

## 2. R_rep construction

R_rep reuses the C/S/R math of `src/metrics.py` verbatim (WS3 floor included). It changes **(a)** how a direction is extracted from the candidate, **(b)** who certifies the edit, **(c)** how methods aggregate.

### 2.1 Architecture-neutral direction extraction (`d_c`)

**Problem the first draft missed.** The mean input-gradient of the logit-difference is *exact and noise-free* for `LinearProbe` (constant in `x`, equals `w[1]−w[0]`) but a *noisy local linearization* for MLP/MKA. Regression dilution then attenuates `|cos(d_c, v_c)|` for the non-linear archs, biasing both ground-truth recovery and R_rep **pro-linear** — a systematic cross-arch confound (both `A` and R_rep would then correlate through arch, not concept).

**Fix — one estimator, equalized noise, for all three archs.** Extract the direction as the **top-`r` right-singular vectors of the per-example logit-difference Jacobian** over a **fixed reference set** `X_ref` of registered size `N_dir` (same `X_ref` for every candidate, so estimation noise is matched across archs):

```
# G[n,:] = ∇_x ( logit_1(x_n) − logit_0(x_n) )   over x_n ∈ X_ref   (grad-ENABLED forward, NOT probe_logits @no_grad, probes.py:328)
# For LinearProbe every row of G is identical == w[1]-w[0]; SVD returns it exactly (rank 1).
U, S, Vh = torch.linalg.svd(G, full_matrices=False)   # Vh[:r] are the top-r directions in R^D
D_c = Vh[:r].T                                        # (D, r); r=1 default (§8 D3)
# guard degenerate directions:  if S[0] < GRAD_NORM_MIN  ->  d_c = None (distinct from 'small but real')
Q, _ = torch.linalg.qr(D_c)                           # orthonormal basis of the candidate subspace
```

- `r = 1` recovers a single unit direction; `r > 1` is now well-defined (fixes the first draft's dead `rank` param). `D_c` is `(D, r)`; the projector is subspace-valued (§2.2).
- Report, per arch, the **bootstrap SD of `|cos(d_c, v_c)|` over resamples of `X_ref`** so attenuation is visible; run the **linear-concept calibration** of §6.6 (all archs must reach equal recovery within CI on a purely-linear concept) before any cross-arch claim.
- Assert `logits.shape[1] == 2` (binary v2 scope; fail loudly if a `>2`-class task is ever added).
- Cast reps to `.float()` mirroring the pipeline (`probes.py:288,321,335`).

### 2.2 Edit: INLP-style subspace erasure derived from the candidate

```
P_erase = I_D − Q Qᵀ            # rank-r projector removing the candidate subspace
X_post_cand = X_inter @ P_erase # one matmul; cache and reuse across evaluators
```

This is the analogue of `inlp_projection` (`interventions.py:34–75`) with the deflation subspace supplied by the **candidate** instead of a freshly trained internal probe. We deliberately do **not** reuse AlterRep/FGSM/PGD (they re-inject probe params/gradient additively and re-open F1 without adding information).

### 2.3 Independent certifiers — data-independent, bagged, quality-gated

**Problem the first draft missed.** "Independence via `set_seed`" is near-useless for convex linear certifiers: `train_probe` varies only init + minibatch order on the *same* fold (`probes.py:283`, `torch.randperm(n, device=X.device)` — also CUDA-stream dependent), so `E` evaluators converge to near-identical discriminants and `sd_e → 0`, which then *spuriously fails* battery check (i).

**Fix — independence from data, not seed.**
- **4-way disjoint split** (§7.4): `X_cand` (candidate training, full — unchanged capacity), `X_eval` (evaluator training), `X_inter` (edit substrate + C/S measurement), `X_test` (candidate accuracy). No fold trains both the candidate and an evaluator; no evaluator trains on the edit substrate.
- **Bagging:** evaluator `e` trains on a bootstrap resample (registered fraction `BAG_FRAC`, §8 D1) of `X_eval`, so `w_e` genuinely varies. `set_seed(EVAL_SEED_BASE + e)` (concrete `EVAL_SEED_BASE = 900000`, provably disjoint from candidate seeds `1000+k` at `run_benchmark.py:479` and gate seed 777 at `gates.py:122`) controls only reproducibility.
- **Quality gate:** each evaluator's held-out `acc_zc`/`acc_ze` (on a slice of `X_eval` **not** in its bag — the v1 `train_validation_probes` accuracy is in-sample, `metrics.py:104`, so we recompute out-of-sample) must clear `EVAL_MIN_ACC` (§8 D10). A blind evaluator **aborts the cell**, distinct from a candidate legitimately erasing the concept (low R). Runtime assert: pairwise `cos(w_e, w_{e'}) < 1 − ε` and that `set_seed` reseeds CUDA.
- Certifiers are `LinearProbe` (sensitive, low-capacity readouts; `metrics.py:50–53`) **by default**, plus **one MLP certifier pair** reported alongside (§4.2 C-b′) to quantify any linear-lens artifact.

`E = 5` default (§8 D1).

### 2.4 Floors (WS3, unchanged)

Each `(candidate, evaluator e, method m)` yields four accuracies via
`compute_intervention_metrics(val_probes_e, X_pre=X_inter, X_post=X_post_m, zc=zc_inter, ze=ze_inter, device, floor=FLOOR)` (`metrics.py:241`), which nulls `C/S/R` when `acc_zc_pre < FLOOR` / `acc_ze_pre < FLOOR` (`metrics.py:216–219`) and nulls the ratio on degenerate denominators (`_safe_ratio`, `|room|<1e-9`, `:175`). Read `m.reliability` (None-aware), never `m.reliability_clipped`. `FLOOR = 0.55` default, `> chance` required (`:196`). **Null accounting is a first-class output** (§5, §6.4): informative missingness (hard cases going null) silently biases every rank correlation.

### 2.5 Aggregation — mean over evaluators, decomposition ladder over methods; NEVER max

Per-evaluator per-method reliability `R_{e,m} = harmonic_mean(C_{e,m}, S_{e,m})` (None below floor). Aggregate over evaluators first (mean of `E` independent certifications), then present methods as a **ladder**, not a `max`:

```
R̄_m       = mean_e R_{e,m}                 (over evaluators with defined R; None if all E null)
sd_m       = sd_e R_{e,m}                   (within-cell certifier dispersion -> battery (i))
```

| Rung | Definition | Isolates | Circularity exposure |
|---|---|---|---|
| `R_v1_max` | `max` over the 5 v1 methods (floored `best_R`, `run_benchmark.py:249`) | the v1 object | F3 present |
| `R_v1_mean` | **mean** over the same 5 methods | the max→mean effect (F3) | F1/F2 present |
| `R_excl` | mean over {INLP, RLACE} only (concept-derived, probe-independent) | dropping self-referential methods | **F1/F2 severed** (edit ⊥ candidate `w`) |
| `R_cand` | the candidate subspace-erasure `R̄_cand` (§2.2) | candidate-specific reliability | F1 maximally present |
| **`R_rep`** | **headline = `R_cand`**, reported **jointly with `R_excl`** | the repair | see §3 |

**Why the ladder, not a single number (corrects the first draft, which made `R_cand` the sole primary and thereby *maximized* F1).** The scientific claim decomposes: `τ(R_v1_max) < τ(R_v1_mean)` is the "max opportunism" mechanism (F3); the increment `R_v1_mean → R_excl` is the exclusion effect (F1/F2); `R_cand` is the candidate-specific object whose reliability the paper ultimately wants `A` to predict. **`R_excl` is the clean, provably-non-circular anchor** (INLP/RLACE never see the candidate — `interventions.py:78,149`). The headline result the paper may claim is exactly whichever of these `A` predicts, reported transparently:
- If `A` predicts `R_excl` (edit direction independent of `w`) → strongest, non-circular by construction.
- If `A` predicts `R_cand` but the synthetic dissociation (§4) holds → non-trivial empirical finding.
- If `A` predicts *only* `R_cand` and dissociation fails → the correlation is the F1 artifact; **claim is not made** (this is the outcome the ydqQ critique warns of, and we commit to reporting it).

INLP/RLACE `X_post` are **candidate-independent** → computed **once per (layer/α)** and reused across all candidates (fixes the per-call `steps=500` RLACE cliff, `interventions.py:120`). Note INLP deflates row `w[0:1]` while `d_c` uses `w[1]−w[0]`; these are different conventions — R_excl and R_cand are different objects **by design**, not accidentally conflated.

**Per-cell record additions** (extends `run_one_cell` return, `run_benchmark.py:257`):
```
"R_rep": R_cand, "R_excl": R_excl,
"R_v1_max": best_R, "R_v1_mean": <mean-5>,      # ladder rungs for the head-to-head
"R_rep_per_method": {m: R̄_m}, "R_rep_sd": {m: sd_m},
"R_rep_n_eval_defined": {m: n_m},               # missingness accounting
"A": A, "lambda_max": lambda_max,               # A never enters any R_* above
```

---

## 3. Why R_rep is non-circular — argument + the decisive check

### 3.1 The argument (honest version)

1. **F3 severed** by banning `max` and reporting the ladder (`R_v1_mean` vs `R_v1_max` quantifies it).
2. **F2 severed** by certifiers that are data-disjoint from the candidate (different fold, bagged) and never see `X_post` during training.
3. **F1 severed for `R_excl`** (edit direction from INLP/RLACE, provably `⊥` candidate `w`) and **for MLP/MKA `R_cand`** (`d_c` is a Jacobian-SVD functional, not any stored weight row). **F1 is NOT severed for linear `R_cand`** — we do not pretend otherwise. For the linear arch the non-circularity of `R_cand` rests entirely on point 4.
4. **The decisive empirical fact (dissociation).** On the environment-shift generator (§6) we *construct* probes that are confident on a spurious shortcut: high training confidence ⇒ **high `A`** (A is computed on training reps/Hessian), but their `d_c ≈ v_s` (shortcut axis), so erasing `d_c` does not remove the true concept as read by certifiers trained where the shortcut is decorrelated ⇒ **low `R_rep`**. Because `A` and `R_rep` are **constructively dissociable**, an empirical `A↔R_rep` correlation on real data is not an identity. This — not any normalization argument — is what answers ydqQ.

### 3.2 Falsification controls (each has a fail rule that sends the metric back)

- **C-a — Direction-source ablation (primary falsification, promoted from diagnostic).** Compute R_rep with (E1) the candidate's own `d_c` and (E2) `d_c^indep` = discriminant of an **independently-trained probe on `X_eval`** (never the candidate's `w`). Report `corr(A, R_cand^E1)` vs `corr(A, R_cand^E2)`. **If the correlation survives under E2** (direction provably independent of `w`), `A` tracks genuine reliability. **If it collapses under E2**, the E1 correlation was the shared-`w` artifact → circular → do not claim. (This subsumes and strengthens the first draft's weak "params-swap" C-a; it also injects an external direction, so `repaired_reliability` takes a `d_override` param — §7.)
- **C-b — A-ablation (CI unit test).** Static assertion + test that R_rep is bit-identical when `A` is set to NaN. No accidental leakage of predictor into target.
- **C-b′ — Certifier-lens ablation.** Report R_rep under linear vs MLP certifier; divergence bounds the linear-subspace artifact (answers "certifiers share the linear lens").
- **C-c — Independence stress (2×2, not 1-D).** Cross `sample-disjoint × geometry-disjoint`: (i) evaluators on the same fold+seed as candidate (breaks sample independence) → R_rep should inflate and its `A`-corr strengthen; (ii) evaluators trained on a **different layer** that still contains `v_c` (geometry-disjoint) → R_rep should be *stable* if the concept, not a shared confound, is certified. Fail if (i) shows no inflation (independence is a no-op) or (ii) swings wildly (certification rode shared-geometry confounds).
- **C-d — Shuffled-candidate collapse** (also battery iii): shuffled-label candidate ⇒ `R_rep ≈ 0` even if `A` arbitrary.

---

## 4. Validation battery (registered; all run on the synthetic generator)

All thresholds are `PREREGISTRATION_v2.md` placeholders (§8). **The battery must pass on a generator family it was NOT calibrated on** (calibrate thresholds on the linear family; evaluate on the non-linear + non-orthogonal family, §6.6) — otherwise it only tests self-consistency of the calibration.

- **(i) Reliability of the estimator (reframed from "variance tracks quality").** The metric must distinguish candidates: intraclass correlation `ICC = var_between_candidates / (var_between + SE²)` with `SE = mean_c sd_e(R_cand)/√E` must satisfy `ICC ≥ ICC_MIN`. PASS iff `ICC ≥ ICC_MIN` and `SE` non-degenerate. (Replaces the unmotivated "worse→more variance"; bagging §2.3 is what makes `sd_e` real.)
- **(ii) No saturation.** `sat_frac = P(R_rep ≥ 1−δ) ≤ SAT_MAX` over the top-quality band **and** dynamic range `P95−P5 ≥ RANGE_MIN`.
- **(iii) Shuffled-label candidates collapse.** `P95(R_rep^shuf) ≤ SHUF_MAX` (reuse the gate shuffle idiom, `gates.py:161`). Register the expected `A^shuf` distribution (should be low) so quadrant bookkeeping in (v) is interpretable.
- **(iv) Random-direction floor, D-relative.** `d_rand ~ Uniform(S^{D−1})`; `P95(R_rep^rand) ≤ RAND_MAX` where **`RAND_MAX` is set relative to the analytic random-cosine null `E|cos(d_rand,v_c)| ≈ √(2/(πD))`** (≈0.05 at D=256), not an absolute constant, and margin `R_rep^good − P95(R_rep^rand) ≥ MARGIN_MIN`. Report `ρ_rec` against this correct chance floor (recovery of 0.05, not 0).
- **(v) Confident-shortcut dissociation (the crux of ydqQ; new, decisive).** In the environment-shift generator, train candidates that latch the shortcut: register `R_rep ≤ SHORT_MAX` while `A ≥ A_HIGH`, with the full 2×2 `{high/low A} × {high/low R_rep}` populated across the α grid. FAIL ⇒ `A` and `R_rep` cannot dissociate ⇒ the correlation, if found, is trivial ⇒ the paper cannot claim non-circularity.

Battery output → `results/ws5_battery.json` with locked thresholds echoed.

---

## 5. (removed — folded into §4/§6)

---

## 6. Synthetic study

### 6.1 Environment-shift generative model (identifiable `v_c`)

**Problem the first draft missed (identifiability).** In a single-environment model with `s_n` correlated to `y_n`, the Bayes-optimal discriminant is a *mixture* `β_c v_c + β_s v_s`; `v_c` and `v_s` are **not separately identifiable**, so `ρ_rec = |cos(d_c,v_c)|` drops with α for an ideal probe purely from confounding, and "causal" is undefined.

**Fix — invariant-vs-spurious across two environments.** Draw orthonormal `v_c, v_e, v_s ∈ R^D` (QR of a 3×D Gaussian; D=256 default). For environment `g ∈ {train, eval}` with shortcut-agreement `α_g`:
```
y_n, e_n ~ Bernoulli(½) independent          # label; selectivity attractor (label-uncorrelated)
s_n = y_n           w.p. α_g   else 1−y_n     # shortcut agreement, environment-specific
x_n = μ_c(2y_n−1)v_c + μ_e(2e_n−1)v_e + μ_s(2s_n−1)v_s + σ ξ_n,   ξ_n ~ N(0,I_D)
```
- **`α_train` swept in `{0.5,…,0.98}`** (§8 D5); **`α_eval = 0.5`** (shortcut decorrelated where certifiers train and edits are scored). Then `v_c` is the **invariant** predictor and `v_s` is spurious. "Recovery of the causal axis" is now operational: a shortcut-latching probe fails in `eval`.
- **Ground truth** `ρ_rec = |cos(d_c, v_c)|` is now clean (invariant axis is identifiable across environments). Also report contamination `κ = |cos(d_c, v_s)|` and the closed-form per-environment Bayes discriminant (stated in the module docstring for provenance).
- **Secondary condition — correlated, non-orthogonal shortcut** `⟨v_s,v_c⟩ = ρ ≠ 0` (§8 D6): C and S trade off as on real data (erasing `v_c` partly erases `v_s`); confirms the mean aggregation still separates good from bad.
- **Constant-null control (missingness):** the generator sets separability directly, so `μ` are tuned to hold `acc_zc_pre` and `acc_ze_pre` **constant across α** (clearing FLOOR by a registered margin), so floor-nulls do not correlate with the manipulation (§6.4). A separate deliberate near-FLOOR condition tests the floor itself (WS3/WS2).

### 6.2 Non-linear family (held-out; where linear-lens circularity would hide)

A purely-linear generator makes the linear discriminant the only recoverable object, so it cannot exercise MLP/MKA capacity or expose linear-lens circularity. Add a **non-linear planted concept** family: `y = XOR(sign⟨x,v_a⟩, sign⟨x,v_b⟩)` (or radial), ground truth = recovery of the **2-D subspace** `span(v_a,v_b)` (principal-angle cosine), with **MLP certifiers**. Thresholds are calibrated on the linear family and **evaluated** here (§4 preamble; Finding-9 fix). Report the head-to-head per-family.

### 6.3 Injection into the pipeline (bypassing extraction)

Planted reps have no sentences, so we construct the dict shape `extract_all_layers` returns and feed the identical downstream sequence **in memory** (no on-disk cache — synthetic "examples" lack the stable string identity `hash_examples` keys on, `repro.py:43`):
```
reps = { L: {"X": X_L(float32), "zc": zc(long), "ze": ze(long)} }   # per environment
```
Then `learnability_gate(...)` (`gates.py:109`) as a WS2 precondition, and `E` bagged evaluators on `X_eval` (§2.3). A thin `SyntheticTask(Task)` is registered in `_TASK_REGISTRY` (`tasks.py:364`) for label bookkeeping only (`name='synthetic'`, `chance=0.5`, gate floors); rep production is bypassed. Populate all four `(zc,ze)` cells and **verify 4-cell balance empirically per replicate** (with a registered tolerance) so no replicate silently drops to the zc-only split fallback (`tasks.py:106`).

### 6.4 Per-arch candidate training and ground-truth recovery

For each `(arch∈{linear,mlp,mka}, α_train, seed, replicate)`: train the candidate on `X_cand` via `train_probe` (`probes.py:261`) with the benchmark `ProbeTrainConfig` (MKA `mka_lambda>0`; MLP is `mka_lambda=0` ≡ MLP — never treat zero-λ MKA as distinct); extract `d_c` (§2.1) on the fixed `X_ref`; compute `ρ_rec`, `κ`; run the full R_rep ladder (§2.5) with bagged certifiers; compute `A` (for the downstream real-data claim — **not** used here). **Null accounting:** persist `n_null/n_total` per `(α, arch)`; require it flat across α (else missingness is confounded with the manipulation).

### 6.5 Head-to-head (correct inference)

**Claim:** R_rep ranks candidates by ground-truth recovery `ρ_rec`; the v1 `max`-R does not, and the gap concentrates where the shortcut is strong.

**Unit of analysis = generator replicate** (NOT the cell). The `18·S` cells within a replicate share `v_c,v_e,v_s`, the `E` certifiers, and the designed α-grid; treating them as iid (as the first draft's power sketch did) is wrong. Inference:
- **P1** `τ_rep = Kendall τ_b(R_rep, ρ_rec) ≥ TAU_MIN`. Null `τ_rep = 0` via permuting `ρ_rec`.
- **P2** `Δτ = τ_rep − τ_v1 ≥ DELTA_MIN`, gated on P1 (so "R_rep barely beats a noise-level v1" is not a pass). `R_v1 := floored best_R` (**never** `R_legacy` — using the clipped v1 field would re-inject the constant-1.0 pathology and manufacture the win). Inference = **cluster bootstrap resampling whole replicates** (each carries all its cells); success = `Δτ` bootstrap-CI lower bound `> 0`. Comparison of dependent correlations (shared `ρ_rec`) → Steiger's test as a cross-check. **Drop the label-swap permutation** (it tests exchangeability of values, not the dependent-correlation hypothesis).
- **P2 must be read off the decomposition ladder** (§2.5): report `τ` for `R_v1_max, R_v1_mean, R_excl, R_cand` so `Δτ` is attributable to the *repair* (max→mean→exclusion→candidate-projection), not to "projection is a nicer edit than FGSM."
- **P3 (mechanism, as a test not a plot):** fit `τ_v1(α)` and `τ_rep(α)` with replicate random effects; register `slope_α(τ_v1) < 0` AND `slope_α(τ_rep) ≈ 0` (equivalence test, registered margin). Power the interaction from the pilot.
- **Missingness sensitivity:** report the head-to-head on (a) the "R-defined for both" subset with exclusion counts and (b) worst-rank imputation of nulls; both must agree in sign.
- Partial out candidate **margin/‖w‖ and confidence/curvature** (not just accuracy), so R_rep is not merely re-reading the confidence `A` also reads (§ Finding-5 fix).

### 6.6 Cross-arch neutrality calibration (gate before any cross-arch claim)

On a **purely linear** concept, all three archs must reach equal `ρ_rec` and `R_rep` within CI (Jacobian-SVD estimator neutrality). If MLP/MKA sit systematically lower, the estimator is non-neutral — correct (disattenuation / larger `N_dir`) before proceeding, and **report `τ_rep` within arch**, never only pooled (an arch-confound must not masquerade as an `A↔R` law).

### 6.7 Power (replicate-level)

Unit = replicate. Run a ≥5-replicate pilot, estimate `SD_replicate(Δτ)`, size `R` so `Δτ_expected / (SD_replicate/√R) ≥ 2.49` (one-sided 80% @0.05). Within-replicate `(arch × α × seed)` is precision, not extra N. `S=8` seeds, `R=20` replicates are the **starting** defaults (§8 D7), locked only after the pilot's between-replicate variance is measured. Battery iii/iv/v need ≥200 shuffled/random/shortcut draws each per α band for stable P95.

---

## 7. Module / API plan

New files compose existing public APIs; no edits to `metrics.py`, `probes.py`, `hessian.py`, `interventions.py`.

### 7.1 `src/ws5_repaired.py`
```python
def candidate_direction(probe, X_ref, device, r=1, batch_size=1024,
                        grad_norm_min=GRAD_NORM_MIN) -> torch.Tensor | None:
    """(D, r) orthonormal candidate subspace = top-r right-singular vecs of the
       per-example logit-diff Jacobian over the FIXED X_ref. None if degenerate.
       Grad-enabled forward via probe(xb) (NOT probe_logits @no_grad, probes.py:328).
       Asserts logits.shape[1]==2."""

def candidate_projection(Q) -> torch.Tensor:            # (D,D) = I - Q Qᵀ

def train_independent_evaluators(X_eval, zc, ze, cfg, device, e_seeds,
                                 bag_frac=BAG_FRAC, min_acc=EVAL_MIN_ACC) -> list[ValidationProbes]:
    """One bagged ValidationProbes per seed (bootstrap resample of X_eval).
       Out-of-sample acc gate; aborts on a blind evaluator; asserts inter-evaluator
       decorrelation. Uses metrics.train_validation_probes internally."""

def repaired_reliability(probe, *, X_ref, X_eval_evaluators, X_inter, zc_inter, ze_inter,
                         device, floor=DECODABILITY_FLOOR, rank=1,
                         d_override=None,                # for C-a E2 direction injection
                         concept_edits=None) -> dict:    # precomputed INLP/RLACE X_post (per layer/α)
    """Returns the ladder + per-method means/SD/null-counts. MEAN over evaluators;
       ladder over methods; NEVER max. 'cand' edit = X_inter @ (I - QQᵀ)."""
```
Reuses verbatim: `metrics.compute_intervention_metrics`/`metrics_from_accuracies` (WS3 floor), `metrics.train_validation_probes`/`ValidationProbes`, `probes.train_probe`/`probe_accuracy`, `interventions.apply_inlp`/`apply_rlace`. **AlterRep/FGSM/PGD are never called.**

### 7.2 `src/ws5_synthetic.py`
```python
@dataclass
class SyntheticConfig:
    D=256; N=8000; family="linear"          # or "nonlinear"
    alpha_train=0.7; alpha_eval=0.5; vs_vc_corr=0.0
    mu_c=1.0; mu_e=1.0; mu_s=1.0; sigma=1.0; seed=0

def make_planted_reps(cfg) -> tuple[dict, dict, dict]:   # (reps_train, reps_eval, truth{v_c,v_e,v_s,bayes})
def recovery_score(d_c, v_c) -> float                    # |cos| (or principal-angle cos for subspace)
class SyntheticTask(Task): ...                            # registry bookkeeping only
```
Reps injected in memory; MKA forwards stay mini-batched (O(N²) kernels, `probes.py:94,164`).

### 7.3 `scripts/ws5_run.py` (driver)
Loops `(family, arch, α_train, seed, replicate)`; per replicate: gate once, build `E` bagged certifiers once, compute concept edits (INLP/RLACE) once; per candidate: `ρ_rec`, the R_rep ladder, `A`. Emits schema-v2 JSONL (git commit stamped like `run_benchmark.py`), `results/ws5_battery.json` (checks i–v, locked thresholds), `results/ws5_headtohead.json` (τ ladder, replicate cluster-bootstrap CIs, Steiger, interaction test, missingness sensitivity).

### 7.4 Real-data path — this is a benchmark change, not a "6-line patch"

**Honest correction.** R_rep needs a **4-way disjoint split** `{X_cand, X_eval, X_inter, X_test}`. v1 trains candidate *and* validators on the same `X_probe` (`run_benchmark.py:222,466`); carving an evaluator fold from it would shrink candidate training data and silently change `A`/accuracy for **every** cell. Since v2 is a clean re-run with its own prereg, we adopt the 4-way split as a **registered WS8 data-pipeline change** (split fractions in §8 D11), and we do **not** claim v1↔v2 candidate comparability (there is none — it's a re-run). `run_one_cell` gains the R_rep ladder fields (§2.5); evaluators+concept-edits built once per layer; **R_rep gated behind `--repaired` flag** so the base benchmark isn't slowed for runs that don't need it.

### 7.5 Compute budget (real path, per layer)

Unchanged and dominant: PyHessian power iteration per `(layer,arch,seed)` (`hessian.py`, `max_iter=100,top_n=20`). Added by WS5: `E` bagged evaluator trainings **once per layer**; concept edits (INLP `num_iters`, RLACE `steps=500`) **once per layer**; per candidate: one `X_inter@P_erase` matmul + `E × n_methods × 4` `probe_accuracy` passes (cheap, batched). Net ≈ +30–60% wall over the Hessian-dominated baseline when `--repaired` is on; budget it explicitly in WS8. Synthetic path: `families × 3 archs × |α| × S × R ≈` a few thousand candidates — hours, not days.

---

## 8. Open experimenter decisions (lock in `PREREGISTRATION_v2.md`)

| # | Decision | Recommended default | Evidence to lock |
|---|---|---|---|
| D1 | Evaluators `E`, bag fraction `BAG_FRAC` | `E=5`, `BAG_FRAC=0.8` | Sweep `E∈{3,5,10}`; lock smallest `E` giving stable `R_rep` **and** non-degenerate `sd_e` (ICC battery-i passes). Bagging, not seed, supplies independence. |
| D2 | Headline metric | report **both `R_excl` (non-circular anchor) and `R_cand`**; claim only what survives §3.2 C-a(E2) + §4(v) | Lock `R_excl` as primary if `A↔R_excl` holds; else `R_cand` only with dissociation passing. |
| D3 | Direction rank `r`, `N_dir`, `GRAD_NORM_MIN` | `r=1`, `N_dir=4000`, guard on smallest generator margin | Compare `r∈{1,2,4}` vs `ρ_rec`; lock `r=1` unless higher `r` improves `τ_rep` without saturation (ii). |
| D4 | Decodability floor `FLOOR` | `0.55` (WS3) | Same lock as WS3; verify null flips at `acc_zc_pre = 0.55±0.03`. |
| D5 | `α_train` grid, `α_eval` | `{0.50,0.60,0.70,0.80,0.90,0.98}`, `α_eval=0.5` | Confirm grid spans a regime where `τ_v1` degrades (P3); extend toward 0.99 if flat. |
| D6 | Signal knobs `μ_c,μ_e,μ_s,σ`, shortcut corr `ρ` | `1,1,1,1`; `ρ∈{0,0.3}` | Tune on a 100-cell pilot so mid-α `acc∈[0.75,0.9]`, above floors/below saturation, null-rate flat across α. |
| D7 | Candidate seeds `S`, replicates `R` | `S=8`, `R=20` (start) | Lock from pilot **between-replicate** `SD(Δτ)` (§6.7), not the iid formula. |
| D8 | Battery thresholds | `ICC_MIN=0.7, δ=0.02, SAT_MAX=0.20, RANGE_MIN=0.30, SHUF_MAX=0.10, RAND_MAX = 2·√(2/(πD)), MARGIN_MIN=0.30, SHORT_MAX=0.10, A_HIGH`=pilot-P75(A) | Calibrate on the **linear** family; **evaluate on the non-linear/non-orthogonal family** (must still pass). |
| D9 | Head-to-head statistic | Kendall `τ_b`; replicate cluster-bootstrap CI (B=2000); Steiger cross-check; P3 interaction test | Success = `Δτ` CI lower bound `>0`, `τ_rep≥TAU_MIN=0.5`, P3 slopes as registered. |
| D10 | `EVAL_MIN_ACC`, evaluator OOS target | `0.90` held-out | Verify bagged certifiers clear it at the chosen `X_eval` size; else enlarge `X_eval`. |
| D11 | 4-way split fractions `{cand,eval,inter,test}` | `{0.40,0.20,0.30,0.10}` | Candidate must retain enough to train to the v1 accuracy band; evaluators must clear D10. This changes the WS8 data pipeline. |

`EVAL_SEED_BASE = 900000` (fixed constant, disjoint from candidate `1000+k` and gate `777`).

---

## 9. Sign-off checklist

- [ ] D1–D11 locked in `PREREGISTRATION_v2.md`.
- [ ] Cross-arch neutrality calibration (§6.6) passes: equal `ρ_rec`/`R_rep` across archs on a linear concept.
- [ ] Battery (i)–(v) PASS on the **held-out** (non-linear/non-orthogonal) family, including (v) the confident-shortcut dissociation with all four `A×R_rep` quadrants populated.
- [ ] Falsification controls C-a(E1 vs **E2**), C-b, C-b′, C-c(2×2), C-d all pass; C-a's E2 result decides whether the headline claim is `R_excl`, `R_cand`, or **not made**.
- [ ] Head-to-head: replicate-clustered `Δτ` CI lower bound `> 0`; ladder attributes the gain to the repair; P3 interaction as registered; missingness sensitivity agrees in sign.
- [ ] Only then: compute real-data `A ↔ {R_excl, R_cand}` and report the surviving claim as the non-circular reliability result.

**Files delivered by WS5:** `src/ws5_repaired.py`, `src/ws5_synthetic.py`, `scripts/ws5_run.py`, the WS5 section of `PREREGISTRATION_v2.md`, a `--repaired` extension to `scripts/run_benchmark.py:run_one_cell` (with the WS8 4-way split), and a one-line `_TASK_REGISTRY` addition. No changes to `src/metrics.py`, `src/probes.py`, `src/hessian.py`, `src/interventions.py`.
