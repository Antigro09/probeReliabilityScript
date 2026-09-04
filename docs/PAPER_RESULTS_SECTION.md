# Draft results section (v2) — for the paper

Camera-ready-ish prose for the results section. All numbers are from the locked
v2 run (`prereg-v2` + the Gemma left-padding fix) and the full synthetic study
(linear R=20/S=8, n=2,880; held-out non-linear family with MLP certifiers,
n=540). Figures: `results/figures/fig0` (methods pipeline) and `fig{1,2,3,4}`.

---

## A repaired reliability metric

We first show the v1 reliability target is not a property of the probe being
evaluated. In the v1 benchmark, the validation probes and the five interventions
are computed once per (model, layer, task) cell and never depend on the candidate
probe, so the reliability R is identical across all architectures and seeds within
a cell. A predictor of R therefore has nothing per-probe to predict, and the v1
"rank-1 hit rate" is a forced tie. We repair this by deriving the intervention
from the candidate probe's own extracted direction (the top right-singular vector
of its per-example logit-difference Jacobian) and certifying the effect with
E = 5 independent, bagged evaluator probes trained on a disjoint clean fold. The
resulting metric, R_cand, is aggregated by mean over evaluators (never max) with
a decodability floor (below-floor cells are null, not clipped). Across the v2
records, R_cand takes 60 distinct values in 51 of 168 cells (0% of cells are
degenerate), confirming it is a genuine per-probe quantity.

## R_cand is valid: synthetic construct validity

To license R_cand as a reliability measure — and to answer the circularity
concern that a candidate-derived intervention scored by a probe could be
self-fulfilling — we validate it on a synthetic generator with a *known* causal
axis. The generator plants an invariant causal direction v_c, a label-uncorrelated
selectivity axis v_e, and a spurious shortcut axis v_s whose label-correlation
differs between the candidate's training environment and the evaluators'
environment, so that v_c is identifiable as the invariant predictor. Ground truth
is each probe's recovery of v_c. On a linear-Gaussian generator, R_cand ranks
probes by ground-truth recovery at Kendall τ = 0.80 (n = 2,880 candidates), far
above the v1 max-over-methods metric (τ = 0.04); the paired advantage Δτ = 0.76,
95% CI [0.66, 0.85], excludes zero. The reliability battery passes (ICC = 0.99;
shuffled-label candidates collapse below 0.10; random directions score at the
D-dependent chance floor). On a held-out non-linear (XOR) generator with
non-linear certifiers the ranking holds (τ = 0.75, Δτ = 0.70, CI [0.59, 0.83]),
and — decisively — the confident-shortcut *dissociation* check passes: probes
confident on the spurious shortcut (high alignment A) but with no true causal
recovery receive low R_cand (112 such high-A / low-R_cand cells). R_cand is thus
*not* a relabeling of A, so an empirical A↔R_cand correlation would be a genuine
finding rather than an artifact. Crucially, A is never computed anywhere in this
validation, so R_cand's validity does not depend on the hypothesis under test.

An auxiliary attacker/evaluator control confirms the specific confound R_cand is
designed to avoid: scoring an intervention with the *same* probe that generated
it inflates completeness relative to an *independent* evaluator (AlterRep on
BERT/SVA: matched C = 1.00 vs split C = 0.86, gap 0.14), which is why R_cand
certifies every edit with disjoint, independently-seeded evaluator probes.

## Alignment does not predict repaired reliability

With a validated, probe-dependent target, we test the pre-registered predictions
(RP1–RP3) on the benchmark of 6 models × 3 tasks × 5 layers × 3 architectures ×
20 seeds (3360 probes; 168 (model, layer, task, architecture) cells after the
learnability gate). The Spearman correlation between the alignment score
A = mean|cos(w, top-20 Hessian eigenvectors)| and R_cand, over cell medians, is
**ρ = −0.04** (permutation p = 0.59; cluster-bootstrap 95% CI [−0.23, 0.15],
clustered by (model, layer, task)). The tie-aware rank-1 architecture hit rate is
0.38, at chance for three architectures. Both pre-registered thresholds
(ρ ≥ 0.5; hit rate ≥ 0.5) are missed; the registered outcome is **negative**.

The decomposition ladder localizes why (Fig. 2). Against the *v1-style*
max-over-methods reliability, A correlates significantly — but in the *wrong*
direction (ρ = −0.32, p < 0.001); against the repaired, validated R_cand it is
null (ρ = −0.04). The apparent v1 signal was thus an artifact of a target that
did not depend on the probe, and where a v1-style target does vary, higher
alignment predicts *lower* apparent reliability. At the architecture level,
alignment and reliability move oppositely (Fig. 3): higher-capacity probes
(MLP, MKA) are more reliable by R_cand (median 0.48 vs 0.40 for linear) yet less
aligned (median A 0.004/0.001 vs 0.007), which is why the rank-1 hit rate fails.
Finally, probe test accuracy anti-correlates with reliability
(ρ(accuracy, R_cand) = −0.19, p = 0.014): more accurate probes are *less*
causally reliable, consistent with high-fit probes exploiting shortcuts whose
directions are less causally clean.

## Interpretation

Hessian directional alignment does not transfer to operational probe selection
when reliability is measured with a probe-dependent, ground-truth-validated
metric. This is the sharpest form of the negative result: the predictor fails
against a *correct* target, not against the broken one that produced the original
observation. The contribution is therefore (i) a repaired, validated reliability
metric that exposes a latent defect in the standard evaluation, and (ii) the
finding — pre-registered and reported regardless of direction — that curvature/
geometry alignment is not a usable proxy for causal probe reliability.

## Limitations

The gender task (Winogender, full-sentence, last-token) is underpowered at 188
examples and does not survive the learnability gate on most models; we report it
only where it passes and do not draw conclusions from it. Gemma-2-2b required a
padding-side fix to its last-token extraction before it decoded above chance; the
learnability gate refused it until then, so no garbage entered the results. The
alignment score A is small in absolute terms (median 0.003, near the
random-alignment floor for these parameter dimensions); it varies enough to rank
architectures but carries no signal about causal reliability.

---

## For the anonymized mirror (4open.science)

Before submission, the anonymized 4open.science mirror must be updated to v2 and
scrubbed of deanonymizing metadata. Checklist:
- Repo/author strings: contributor usernames, contributor email addresses,
  and local home-directory paths (appear in provenance records and
  cache/provenance sidecars — regenerate or scrub).
- Git history/author identity and any commit co-author trailers.
- The mirror must point only at the anonymized copy; the paper must not link the
  GitHub repo. A reviewer finding the GitHub handle is a desk-reject risk.
- This is a non-code step the authors must perform on the mirror host; the code
  changes above do not touch it.
