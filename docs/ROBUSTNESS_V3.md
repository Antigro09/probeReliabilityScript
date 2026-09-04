# Schema-v3 robustness study

The schema-v3 study tests the claims that Phase 2 could not settle. It leaves
the preregistered schema-v2 runner, records, and outcome untouched. Treat every
schema-v3 analysis as post-registered unless you timestamp a new protocol
before running it.

## Questions

The study answers five reviewer questions.

1. Does a fixed evaluator lose accuracy because an edit removed the concept,
   or because the edit inverted its decision rule?
2. Can a fresh linear or nonlinear decoder recover the target after the edit?
3. Does the candidate edit beat rank-matched random, distortion-matched random,
   supervised-direction, LEACE, INLP, and the repository's RLACE approximation?
4. Does the Hessian conclusion survive a basis-invariant alignment statistic,
   architecture-size null normalization, and changes in the top-k subspace?
5. Do sample-disjoint direction estimation, grouped splits, and perturbation
   sweeps change the result?

## Five disjoint folds

| Fold | Fraction | Use |
|---|---:|---|
| `candidate` | 0.30 | Candidate training and Hessian |
| `evaluator` | 0.20 | Bagged fixed evaluators |
| `direction` | 0.15 | Edit fitting and cross-fit checks |
| `decoder` | 0.20 | Fresh post-edit decoder training |
| `test` | 0.15 | Final fixed and fresh decoder scores |

The gender task groups masculine and feminine minimal pairs by default. A pair
cannot cross folds. Regenerate `data/gender.tsv` with the updated generator if
you want occupation-held-out or template-held-out splits:

```bash
python -m scripts.generate_gender_data --download
```

The generator writes `data/gender.metadata.jsonl` next to the three-column TSV.
Then pass `--group-mode occupation` or `--group-mode template`.

## Metrics

The runner saves raw accuracy, complemented accuracy, balanced accuracy, AUC,
log loss, and Brier score. It also saves orientation-corrected forms. If an
edit flips every prediction, raw accuracy falls to zero while oriented AUC
remains one. The analysis counts that case as preserved information.

The fixed-evaluator score uses oriented AUC skill above chance. The analysis
reports it as fixed-evaluator damage, not erasure. The fresh-decoder result is
separate. Strong erasure evidence requires target AUC near 0.5 for both fresh
linear and MLP decoders while fresh control decoders remain accurate.

The Hessian section reports:

```text
A_raw       = mean_i |cos(w, v_i)|
A_sub(k)    = ||Q_k^T w||_2 / ||w||_2
```

`A_sub(k)` is invariant to rotations inside the selected eigenspace. The runner
evaluates `k in {1, 5, 10, 20}`, compares each value with the exact random
subspace null for the probe's parameter count, and checks selected eigenpairs
with independent Hessian-vector products.

## Run order

Run the CPU tests first:

```bash
pytest -q
```

Run one model-task smoke test:

```bash
python -m scripts.ws9_robustness_sweep \
  --models pythia --tasks sva --smoke --analyze
```

Run all registered synthetic conditions, including shortcut-concept
correlation 0.3:

```bash
python -m scripts.ws9_synthetic_controls
```

Run the real-data study:

```bash
python -m scripts.ws9_robustness_sweep \
  --save-predictions --gender-pre-pronoun --analyze
```

Each model-task runs in a subprocess. You can interrupt and resume the command.
The candidate writer resumes by `(layer, architecture, seed)`. Layer exclusions
remain in `*.layers.jsonl`; the analysis includes them in
`analysis/inclusion_matrix.csv`.

The full study trains many fresh decoders and performs second-order residual
checks. Start with one non-gated model. Confirm disk use and wall time before
launching all six models.

All candidates receive fixed-evaluator and matched-control scores. The first
five candidate seeds per architecture form the deep subset and receive fresh
post-edit decoders, cross-fitting, and direction-resampling checks. The runner
checks Hessian eigenpair residuals on seed 1000 in each architecture-layer
cell. These sampling rules keep the final study tractable and appear in every
manifest.

## Outputs

`results/robustness_v3/` contains:

| File | Contents |
|---|---|
| `*.candidates.jsonl` | Candidate metrics, Hessian checks, matched controls |
| `*.layers.jsonl` | Inclusion status, shared baselines, perturbation sweeps |
| `*.manifest.json` | Commit, config, data hash, split counts, gate result |
| `predictions/*.jsonl.gz` | Optional per-example evaluator probabilities |
| `ws9_sweep_summary.json` | Subprocess status and timestamps |
| `analysis/*.csv` | Candidate, baseline, inclusion, and sensitivity tables |
| `analysis/*.png` | Paper figures in the existing gray/salmon/crimson palette |
| `analysis/robustness_summary.json` | Block-bootstrap and sensitivity summary |

Do not copy numbers into the manuscript until all requested model-task cells
have either completed or appear in the inclusion matrix with a reason. Keep
pre-pronoun gender failures in the record. A gate failure there shows that the
visible pronoun supplied the decodable label; it is evidence about leakage, not
a failed production run.

## Claims allowed by each outcome

| Fixed evaluator | Fresh target decoder | Fresh control decoder | Interpretation |
|---|---|---|---|
| damaged | strong | strong | Evaluator damage; target remains decodable |
| damaged | chance | strong | Evidence consistent with selective erasure |
| damaged | chance | chance | Non-selective information loss |
| unchanged | strong | strong | Edit did not remove target information |

Code cannot guarantee a review score. These runs address the current technical
objections. The paper should report the measured outcome, including a negative
or mixed result.

## External pipeline audit

The repository still needs a named released pipeline if the paper claims that
the Phase-1 misuse occurs outside this project. Select that pipeline before
running an external audit. Until then, describe Phase 1 as a benchmark-failure
case study and avoid prevalence claims.

After exporting one row per candidate, run:

```bash
python -m scripts.audit_external_results \
  --input external/results.jsonl \
  --cell-fields model,layer,task \
  --candidate-fields architecture,seed \
  --target-field reliability \
  --per-method-field per_method \
  --out external/audit.json
```
