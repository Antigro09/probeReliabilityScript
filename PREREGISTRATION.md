# Pre-Registration: Hessian Directional Alignment as a Probe Reliability Predictor

**Locked:** May 1, 2026
**Authors:** Anonymous (workshop submission)
**Status:** Predictions made before running the benchmark experiment described below.

---

## Background

Probe reliability is conventionally evaluated by causal interventions on
representations (INLP, RLACE, AlterRep, FGSM, PGD), with reliability R
defined as the harmonic mean of completeness and selectivity (Canby et al.
2024). Running these interventions is computationally expensive, requires
careful task design (defining a target Zc and a non-target Ze), and is
not feasible for every probing decision a practitioner makes.

We have observed in preliminary experiments on Pythia, GPT-2, BERT, and
LLaMA that the directional alignment between probe parameters and the
top eigenvectors of the probe-loss Hessian appears to track failure
modes. Specifically, low alignment with high-curvature directions
co-occurs with intervention reliability collapse in Pythia layer 12 and
GPT-2 layer 12. This is a correlational observation on four data points
and could be coincidence.

We pre-register the following predictions to test whether directional
alignment is **operationally** useful as a probe selection criterion --
that is, whether ranking probes by alignment recovers the same ordering
that ranking by causal reliability would, without paying the cost of
running interventions.

---

## Benchmark Setup (Locked)

**Models (6):**
- BERT-base-uncased
- GPT-2 (124M)
- Pythia-160M
- Qwen2.5-1.5B
- Gemma-2-2B
- LLaMA-3.2-3B

**Tasks (3):**
- Subject-verb agreement (Linzen et al. 2016) -- syntactic
- Gender agreement (Winogender-style templates) -- syntactic
- SST-2 binary sentiment -- semantic

**Probe architectures (3):** Linear, MLP, MKA-regularized MLP

**Layers per model (5):** linspace(1, n_layers, 5)

**Probe seeds per cell (K = 20):** Each (model, layer, task, architecture)
combination is trained 20 times with different random initialization
seeds, producing 20 probes per cell.

**Total probe trainings:** 6 x 5 x 3 x 3 x 20 = 5,400

**Predictor:**
For each probe, the **predictor score** is

    A = mean( |cos(w_flat, v_i)| for i in top_20_eigvecs )

where w_flat is the probe's full flattened parameter vector and v_i are
the top-20 eigenvectors of the probe's training-loss Hessian, also
flattened in the same parameter ordering. This is a single scalar in
[0, 1] computed without running any intervention.

**Reliability:**
For each probe, R is the maximum reliability across the five
intervention methods, where reliability is the harmonic mean of
completeness and selectivity computed against fixed validation probes
trained on un-intervened representations.

**Statistical units:**
The benchmark produces 6 x 5 x 3 = 90 cells. Within each cell, the 20
seeds are aggregated to produce alignment ranks and reliability ranks
across the 3 architectures (each architecture's score is the median over
20 seeds). The full benchmark produces 90 x 3 = 270 (alignment, reliability)
pairs, plus 90 within-cell rankings of 3 architectures.

---

## Predictions

### Prediction P1 -- Spearman correlation (primary)

Across all 270 (alignment, reliability) pairs from the benchmark,
the Spearman rank correlation between A and R will be:

**rho >= 0.5, with p < 0.01**

We treat rho >= 0.5 as the threshold because:
- Conventional "moderate correlation" cutoff in social sciences
- Not trivially achievable from random rankings (chance rho ~ 0)
- Gives meaningful signal for practical probe selection

### Prediction P2 -- Rank-1 hit rate (secondary)

Across the 90 (model, layer, task) cells, in at least:

**50% of cells**

the architecture with the highest median A also has the highest median R.

Chance baseline for picking among 3 architectures is 33.3%; the
threshold of 50% is meaningfully above chance.

### Prediction P3 -- Per-task generalization (robustness)

Predictions P1 and P2 will hold separately within each of the three
tasks (90 pairs / 30 cells per task). At minimum:
- Spearman rho >= 0.4 within every task
- Rank-1 hit rate >= 40% within every task

This guards against the operational claim being driven entirely by one
task.

---

## Outcome Categories

We commit to the following interpretation rubric, locked before seeing results:

| P1 met | P2 met | P3 met | Interpretation |
|--------|--------|--------|----------------|
| Yes | Yes | Yes | Strong positive result. Alignment is operationally useful as a cheap proxy for reliability. |
| Yes | Yes | No | Mixed. Aggregate signal exists but doesn't generalize across tasks. We report this honestly. |
| Yes | No | * | Aggregate correlation without practical rank-1 utility. Suggests alignment captures *something* about reliability but isn't operational. |
| No | * | * | **Negative result.** We report it. The paper becomes "geometric and curvature signals correlate observationally but do not transfer to operational probe selection -- here is what this tells us about probe geometry." |

**We commit to publishing the result regardless of which row applies.**

The paper's contribution does not depend on the predictions being
confirmed. A confirmed prediction strengthens the operational claim;
a falsified prediction sharpens the geometric understanding. Both are
useful to the field.

---

## What We Will Not Do

The following are explicitly excluded to prevent post-hoc analysis:

1. **No threshold tuning.** rho = 0.5 and 50% are locked. We will not
   "find" a threshold that the data passes.
2. **No model exclusion.** All 6 models are reported even if one is an
   outlier. We may discuss outliers but not remove them.
3. **No task exclusion.** All 3 tasks are reported.
4. **No architecture exclusion.** All 3 probe architectures (Linear, MLP,
   MKA) are included in the rankings.
5. **No layer exclusion.** All 5 layers per model are reported.
6. **No re-running with different seeds** if the benchmark gives a result
   we don't like.
7. **No alternative predictor formulations** swapped in if A doesn't
   correlate. We use mean alignment with top-20 eigenvectors as defined
   above. If we want to study other formulations, they are reported as
   exploratory analyses, clearly labeled, and do not retroactively
   become the registered predictor.

---

## Pre-Specified Exploratory Analyses

The following are NOT predictions; they are analyses we plan to run for
descriptive interest, with their exploratory status disclosed:

- Per-architecture analysis (does alignment predict better for some
  probe architectures than others?)
- Per-layer-depth analysis (early vs. late layers)
- Alignment with bottom-20 eigenvectors as a complementary diagnostic
- lambda_max magnitude alone as a baseline predictor (we expect this to be
  uninformative based on preliminary results)

---

## Operational Claim (If Predictions Are Met)

If P1, P2, and P3 are all met, the paper claims:

> Probe reliability can be approximated by directional alignment between
> probe parameters and the top eigenvectors of the probe-loss Hessian.
> The alignment score is computable in O(probe_size) per power-iteration
> step using the probe's loss Hessian alone, requires no representation
> interventions, and ranks probes consistently with full intervention-based
> reliability evaluation. We propose alignment-based ranking as a
> compute-efficient probe selection criterion.

If predictions are not met, this claim is not made.

---

## Hash & Signing

This document is committed to the repository at the git revision tagged
`prereg-v1` before any benchmark experiments are run. The git hash of
this commit is the pre-registration's binding identifier.

`scripts/predictor_eval.py` contains the locked computation of A, R,
rho, and the rank-1 hit rate. It is committed alongside this document.
The version of that file at `prereg-v1` is what determines whether the
predictions are met.
