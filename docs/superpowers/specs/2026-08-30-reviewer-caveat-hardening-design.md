# Reviewer Caveat Hardening Design

## Objective

Reduce the reviewer revision's fixable scientific and reproducibility limitations without rewriting the completed registered run or weakening its negative findings. The extension adds a full-case sensitivity analysis for denominator-floor exclusions, prospectively evaluates fresh-decoder recovery in the 11 untouched middle-layer model--task cells, makes the current environment and artifact inventory independently verifiable, updates the manuscript from validated outputs, and pushes a code-only reproduction branch.

## Scientific Boundaries

The completed run `20260830T024520Z-7aab3eb` remains immutable and is the registered primary analysis. The extension must not:

- relabel the epsilon-sweep result, candidate-diagnostic failure, or fixed-decoder limitation as successes;
- alter the locked normalized-damage primary estimand or its denominator floor;
- treat layers, candidates, or edits that share examples as independent model--task blocks;
- describe a task decoder as downstream language-model behavior;
- call the already inspected Qwen/SST-2/layer-14 construct result confirmatory; or
- claim that any finite decoder battery proves concept erasure.

The Qwen/SST-2/layer-14 cell is a disclosed pilot. Thresholds and inference rules below are locked before inspecting construct endpoints in the remaining 11 cells. Results from those cells may strengthen or weaken the paper's interpretation; the implementation reports either outcome unchanged.

### Prospective inference amendment (2026-08-31)

Before any confirmatory endpoint was computed or inspected, the extension registration was clarified to remove ambiguities found during implementation review. This dated amendment does not change the cells, endpoints, thresholds, draw count, seed, or pilot designation. It locks `alpha = 0.05`, NumPy `PCG64`, NumPy quantiles with `method="linear"`, endpoint tail p-values `(1 + count(theta_star <= threshold)) / (B + 1)`, the within-cell intersection-union p-value as the maximum of the three endpoint p-values, and Holm adjustment across exactly the 11 confirmatory cells. A cell passes only when all three point thresholds pass and its Holm-adjusted cell p-value is at most `0.05`; point-threshold and inferential decisions are reported as separate booleans. A nonestimable confirmatory cell remains in the 11-cell family with internal p-value `1` and label `nonestimable`. The pilot remains descriptive only and receives neither a confirmatory p-value nor a confirmatory decision. Every reported one-sided 95% lower bound is marginal, not Holm-adjusted and not simultaneous.

## Scope

### 1. Denominator-floor robustness

The extension derives three post-hoc sensitivity analyses from the saved matched/split AlterRep rows while retaining normalized target damage `C` as the registered primary endpoint.

1. **Full-case raw accuracy-drop contrast.** For each condition, define raw target drop as `target_acc_pre - target_acc_post`. For every attacker/evaluator pair, compute matched raw drop minus split raw drop on the identical edit. Aggregate pairs within model--layer--task cells and selected layers within equally weighted model--task blocks. Report matched and split means, the equal-block gap, a 10,000-draw hierarchical bootstrap over model-task, layer, and pair, and the exact two-sided sign-flip test over nonzero model--task block gaps. This estimand includes all 300 pairs and 60 cells because it has no unstable chance-normalized denominator.
2. **Floor curve.** Recompute the target-only normalized-damage contrast at floors `0.5000000001`, `0.525`, `0.55`, `0.575`, and `0.60`. At each floor, exclude a pair symmetrically if either condition's target denominator is below the threshold, retain every row and reason, and report pair/cell counts, equal-block gap, and exact block sign-flip p-value.
3. **Partial-identification bound.** At the locked `0.55` floor, bound each missing paired normalized-damage gap in `[-1, 1]`, aggregate using the registered hierarchy, and report the resulting worst-case interval. This is a deterministic sensitivity bound, not a confidence interval.

These analyses are labeled post-hoc robustness checks in JSON, figures, and prose. They cannot replace the registered available-case result.

### 2. Prospective 11-cell construct replication

A new immutable extension spec enumerates the middle sampled layer for all 12 model--task blocks. The existing Qwen/SST-2/layer-14 pilot is referenced but never rerun as confirmatory evidence. The other 11 cells are the confirmatory panel.

Each untouched cell runs the existing candidate-conditioned and AlterRep workflow with the same sentence-group-disjoint candidate, evaluator, direction-fit, fresh-decoder-fit, orientation-calibration, and final-test subdivisions. Every artifact is namespaced by model, task, and layer so splits, checkpoints, directions, edited states, and baselines cannot collide across cells. Hyperparameters are selected on unedited decoder-fit data within each cell and frozen across its edits.

The worker persists normalized group sufficient statistics, not precomputed group-level accuracies or recovery ratios. Every row contains the full cell identity, `evaluation_family`, `decoder_seed`, target/control label, `group_id`, `n_examples`, `correct_count`, provenance hashes, and status. Baseline rows are uniquely keyed by `(cell, evaluation_family, decoder_seed, label, group_id)`. Edited rows add `edit_id`, `edit_object`, architecture, candidate seed, and edit hash, and are keyed by `(cell, edit_id, evaluation_family, decoder_seed, label, group_id)`. Each estimable cell must contain the exact 60 candidate edits. Analysis reconstructs example-weighted pooled accuracies and ratios from these counts inside every bootstrap draw.

The primary decoder family is the deterministic fresh linear decoder; fresh MLP is confirmatory robustness. For each untouched cell, material target recovery requires all of the following:

- median fresh-linear post-edit target accuracy across the 60 candidate-conditioned edits is at least `0.55`;
- median fresh-linear target recovery ratio is at least `0.50`, retaining at least half of the unedited above-chance signal; and
- median fresh-linear control-retention ratio is at least `0.80`.

Cell-level one-sided uncertainty uses 10,000 `PCG64` group-bootstrap draws over final-test group IDs. Task examples are resampled jointly across models so models evaluated on the same task do not create false independence. Each draw recomputes example-weighted pooled baseline and edited accuracies, recomputes recovery and retention ratios without clipping, and takes the median over the fixed set of 60 dependent candidate edits. A replicate with target baseline accuracy at or below `0.5` is a conservative endpoint-tail failure. The three plus-one endpoint p-values combine by intersection-union (`max`), then Holm adjusts the 11 cell p-values. The one-sided 95% lower bounds use the marginal 0.05 quantile with NumPy's linear method and are explicitly neither multiplicity-adjusted nor simultaneous. MLP results, AlterRep results, orientation calibration, endpoint distributions, and the number of cells satisfying each rule are reported whether or not the primary rule passes. No inference treats the 60 candidate edits as independent.

The confirmatory output is a panel of cell classifications, separate point/inferential decisions, and marginal bounds, not a universal erasure claim. A scientific subthreshold result remains a successful analysis status rather than an execution failure. A nonestimable confirmatory cell remains in the fixed 11-cell Holm family with p-value `1`; the pilot remains descriptive and outside that family. The paper may say that recovery replicated only for the exact cells and decoder families that pass the locked rule.

### 3. Environment preservation

The code exports a sanitized current-environment lock containing:

- Python version, ABI, platform, and architecture;
- exact installed distribution names and versions in canonical order;
- the source `requirements.txt` hash;
- PyTorch build/version and selected deterministic device protocol; and
- hashes of the extension spec and generating commit.

The exporter rejects absolute local paths, editable installs, credentials, and direct URLs rather than leaking machine-specific state. A human-readable lock file contains `name==version` entries where reconstructible; nonstandard builds such as the recorded development PyTorch build remain explicitly documented instead of being presented as generally installable.

This preserves the current extension environment. It does not claim to reconstruct the historically lost original environment.

### 4. Artifact portability metadata

A `package-artifacts` command emits content-addressed manifests without duplicating the local payload:

- **analysis tier:** row tables, summaries, reports, figures, and paper inputs;
- **audit tier:** analysis tier plus per-example predictions, compact edit recipes/source tensors, and required checkpoints; and
- **bitwise tier:** every referenced file in the immutable run.

Each manifest records normalized relative path, byte size, SHA-256, tier, producing run, and aggregate hash. A verifier rejects missing files, empty files, path escapes, duplicate normalized paths, and hash mismatches. Archive/upload backends are outside this change because no external artifact destination was supplied and the disk cannot hold a second 57-GiB copy. The manifest makes a later streamed deposit or DOI verifiable.

### 5. Paper and figures

Analysis and construct-panel summaries generate manuscript macros; numeric prose is never hand-edited. The manuscript adds the full-case floor robustness result and replaces the one-cell construct paragraph/figure with a pilot-plus-confirmatory-panel report. It keeps the registered epsilon interpretation, candidate-diagnostic failure, narrow-benchmark warning, and fixed-decoder/erasure distinction.

Generated figures include:

- the existing matched/split primary display with a compact floor-sensitivity companion;
- the existing epsilon sweep unchanged except for any regenerated styling consistency; and
- a construct panel showing per-cell recovery, marginal one-sided bounds, control retention, and explicit pilot/confirmatory labeling.

The compiled paper must still satisfy the five-main-page workshop limit, anonymous metadata/text checks, citation/reference checks, overfull-box gate, 300-DPI figure gate, and page-by-page visual inspection.

## Architecture and Data Flow

The original spec and completed run remain untouched. A separate extension YAML is validated by a focused extension-config type. The runner first validates and reanalyzes the immutable primary rows, then executes only missing untouched construct cells into a new immutable run directory. Cell workers write atomic namespaced shards. Analysis validates exact cell/edit/evaluator keys before aggregation. Figures and paper consume validated consolidated summaries only.

Reusable responsibilities remain separated:

- `config.py` validates extension cells, thresholds, bootstrap rules, disk reserve, and required outputs.
- `artifacts.py` creates tiered manifests and verifies portable bundles.
- `analysis.py` owns raw-drop, floor-curve, partial-identification, and multi-cell construct summaries.
- `runner.py` orchestrates namespaced per-cell construct execution, resume validation, disk/runtime gates, environment export, and final consolidation.
- `figures.py` renders saved summaries/rows without model loading.
- `paper.py` validates extension claims and patches only allowlisted manuscript regions.

Before the full panel starts, preflight must show projected additional storage plus an 8-GiB reserve below current free space. The run stops cleanly before model work if this gate fails. Resume is bound to the exact extension spec hash and generating commit.

## Testing and Verification

All behavior changes follow red-green-refactor tests. Unit tests cover raw-drop aggregation, floor endpoints, partial-identification bounds, joint task-group bootstrap determinism, Holm adjustment, cell classification at every threshold boundary, multi-cell key validation, collision-free artifact paths, environment sanitization, and tier-manifest verification. Integration tests cover interruption/resume across two toy cells, pilot exclusion, summary regeneration from rows, macros, figures, and code-only publication checks.

Verification gates are:

1. focused reviewer-revision tests;
2. the full repository test suite;
3. Ruff and whitespace checks;
4. a one-cell extension smoke run;
5. the full 11-cell prospective run with artifact/resume validation;
6. fresh analysis, figures, manuscript compilation, and visual inspection; and
7. an anonymity and committed-file audit.

## Code-Only Git Publication

The remote feature branch must contain source code, tests, YAML specs, design/plan documentation, environment-lock tooling, artifact-manifest tooling, and manuscript source/templates needed to reproduce the work. It must not contain generated run directories, CSV/Parquet result tables, checkpoints, arrays, edited tensors, generated figures, rendered pages, compiled PDFs, LaTeX auxiliaries, or the four unrelated pre-existing deletions.

Because the current unpushed history contains a compact result commit, publication uses an isolated clean worktree to construct the code-only branch. First create a local backup ref preserving the full result-bearing history. Then rebuild `agent/reviewer-revision-2026-08` from its existing base with implementation commits only, omitting generated result blobs and generated paper assets. Verify the rewritten branch contains no `results/reviewer_revision_2026_08/` paths or prohibited generated assets anywhere in commits unique to the branch. Push only after a dry run, full verification, and an explicit comparison against `origin/main`.

The approximately 241 GiB of local failed/intermediate/replay artifacts remain local and untracked. Ordinary Git is not used as an artifact store.

## Success Criteria

The work is ready to push only when:

- the registered primary results remain byte-identical;
- the full-case raw-drop, floor-curve, and partial-identification checks regenerate deterministically from saved rows;
- all 11 untouched construct cells complete or have explicit fatal-unit records under the locked extension rules;
- confirmatory claims are generated strictly from the untouched cells and retain shared-task dependence;
- environment and tiered artifact manifests pass independent verification;
- the paper and figures pass numerical, compilation, layout, anonymity, and visual gates;
- the full test/lint suite passes at the final code commit;
- unique remote-bound history contains code and reproduction metadata only; and
- `git push --set-upstream origin agent/reviewer-revision-2026-08` succeeds without pushing local result payloads or unrelated deletions.

## Irreducible Limitations After This Extension

Even a successful extension cannot prove concept erasure, turn the registered epsilon finding into support for magnitude alone, rehabilitate the rejected candidate diagnostic, recover missing historical archives/environment state, or establish broad external validity/downstream language-model behavior. Those limitations remain explicit.
