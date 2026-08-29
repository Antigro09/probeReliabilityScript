# Reviewer Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use test-driven-development for each production change and verification-before-completion before claiming a gate passes.

**Goal:** Execute the locked August 2026 reviewer-revision experiments, preserve a complete auditable artifact trail, generate the three revision figures and manuscript macros, and compile the anonymous NeurIPS paper only after all scientific gates pass.

**Architecture:** Add an isolated `src/reviewer_revision` package and one CLI driver. Reuse the repository's existing representations and intervention primitives, but put all new run state in immutable timestamped directories. Every experiment unit is an atomic keyed shard; analyses and the paper consume only validated materialized rows.

**Tech Stack:** Python 3.13, PyTorch, NumPy, pandas/Parquet, SciPy, scikit-learn, PyYAML, matplotlib, pytest, TeX Live/latexmk.

---

## Task 1: Locked configuration and immutable artifacts

**Files:**

- Create: `src/reviewer_revision/__init__.py`
- Create: `src/reviewer_revision/config.py`
- Create: `src/reviewer_revision/artifacts.py`
- Test: `tests/reviewer_revision/test_config.py`
- Test: `tests/reviewer_revision/test_artifacts.py`

**Red test:** Assert that the supplied YAML expands to exactly 60 full matched/split cells, 36 fallback cells, 2,400 epsilon rows, and the locked 65 construct edits. Assert invalid epsilon grids and duplicate keys are rejected. Assert interrupted writes leave no valid shard, resume accepts only matching exact keys, and absolute paths are sanitized from manifest payloads.

Run: `.venv\Scripts\python.exe -m pytest -q tests/reviewer_revision/test_config.py tests/reviewer_revision/test_artifacts.py`

Expected red state: import failure for `src.reviewer_revision`.

**Implementation:**

- Parse and validate every locked section used by the driver.
- Generate typed tuple keys for matched/split, epsilon, and construct rows.
- Compute canonical JSON and file SHA-256 hashes.
- Create `<output-root>/<UTC timestamp>-<git short sha>/` with exclusive lock, `manifest.json`, `console.log`, and shard directories.
- Write JSON, CSV, Parquet, Torch checkpoints, and NumPy arrays through temporary files followed by `os.replace`.
- Validate a shard before counting its key as complete.

**Green test:** Re-run the focused tests and `git diff --check`.

## Task 2: Scoring, orientation, extraction, split, and cache invariants

**Files:**

- Create: `src/reviewer_revision/scoring.py`
- Create: `src/reviewer_revision/data.py`
- Modify: `src/extraction.py`
- Test: `tests/reviewer_revision/test_scoring.py`
- Test: `tests/reviewer_revision/test_data.py`
- Test: `tests/reviewer_revision/test_attacks.py`

**Red test:** Cover C/S/H and status reasons for valid, floor, chance, NaN, and degenerate inputs; calibration-only sign selection with deterministic ties; last attended token for left and right padding; exact Phase-2 reconstruction hashes; group-disjoint subdivision; exact cache selection in the presence of duplicate historical files; FGSM/PGD epsilon-zero no-op and L-infinity bounds.

Run: `.venv\Scripts\python.exe -m pytest -q tests/reviewer_revision/test_scoring.py tests/reviewer_revision/test_data.py tests/reviewer_revision/test_attacks.py`

Expected red state: missing scoring/data modules and a failing left-padding assertion.

**Implementation:**

- Return a status-bearing `DamageScore` instead of silently clipping invalid cases.
- Fit orientation only on calibration logits and apply the frozen sign to final logits.
- Locate the true final non-padding position from the mask indices.
- Reconstruct task examples and four Phase-2 folds using the archived deterministic logic.
- Derive stable example/group IDs and persist all subdivision memberships.
- Select caches by model/task/layer/tag plus verified data hash; validate length, finite values, variance, label alignment, and cross-layer split identity.

**Green test:** Run focused tests plus `.venv\Scripts\python.exe -m pytest -q tests/test_interventions_direction.py tests/test_robustness.py`.

## Task 3: Deterministic training and frozen fresh decoders

**Files:**

- Create: `src/reviewer_revision/training.py`
- Test: `tests/reviewer_revision/test_training.py`

**Red test:** Train twice on a tiny separable tensor and require identical history, checkpoint hash, logits, and selected hyperparameters. Require group-disjoint inner validation, patience behavior, frozen hyperparameters across edits, and three deterministic MLP seeds.

Run: `.venv\Scripts\python.exe -m pytest -q tests/reviewer_revision/test_training.py`

Expected red state: missing training module.

**Implementation:**

- Train target/control linear probes with explicit seeds and persisted checkpoints.
- Tune the locked linear grid on unedited decoder-fit data only.
- Train the locked MLP configuration for seeds 0, 1, and 2.
- Persist epoch curves, selected hyperparameters, split IDs, environment metadata, and checkpoint SHA-256.
- Expose evaluation functions returning logits and per-example predictions without reorientation.

**Green test:** Run the focused test twice and compare hashes.

## Task 4: Matched/split and epsilon experiment engines

**Files:**

- Create: `src/reviewer_revision/experiments.py`
- Test: `tests/reviewer_revision/test_matched_split.py`
- Test: `tests/reviewer_revision/test_epsilon_sweep.py`

**Red test:** On synthetic caches, require the same detached edited tensor hash in matched and split rows, five paired seeds, explicit failure rows, no duplicate keys, epsilon-zero identity, monotone epsilon ordering, and resume equivalence after a simulated interruption.

Run: `.venv\Scripts\python.exe -m pytest -q tests/reviewer_revision/test_matched_split.py tests/reviewer_revision/test_epsilon_sweep.py`

Expected red state: missing experiment engine.

**Implementation:**

- Reproduce the archived 12-cell aggregate separately and check the three locked values.
- Train five attacker/evaluator pairs per selected model-task-layer cell.
- Generate AlterRep/FGSM/PGD edits once per unit, detach once, hash once, and score the identical tensor under matched and split evaluators.
- Sweep the ten locked epsilon values at each required middle layer; use PGD step size epsilon/5 and ten steps.
- Emit a validated row for every expected condition or an explicit structured failure row.

**Green test:** Run the focused tests and compare one uninterrupted and one resumed materialization byte-for-byte after stable sorting.

## Task 5: Prespecified Qwen/SST-2 construct check

**Files:**

- Extend: `src/reviewer_revision/experiments.py`
- Extend: `src/reviewer_revision/training.py`
- Test: `tests/reviewer_revision/test_construct_check.py`

**Red test:** Require six-way group disjointness; 5 AlterRep plus 60 candidate-conditioned edit IDs; candidate/evaluator independence; cosine agreement for linear directions; calibration-frozen orientation; final-test labels inaccessible before freezing; and per-example fixed/fresh rows for target and control decoders.

Run: `.venv\Scripts\python.exe -m pytest -q tests/reviewer_revision/test_construct_check.py`

Expected red state: construct entry point absent.

**Implementation:**

- Preserve the archived candidate/evaluator/intervention/test folds and subdivide intervention examples into direction, decoder-fit, calibration, and final-test groups using the locked fractions.
- Retrain the five archived AlterRep directions and the 60 prespecified linear/MLP/MKA candidates.
- Save direction vectors, cosine checks, edit hashes, norms, and evaluator checkpoint hashes.
- Select orientation on calibration only and write raw and oriented final metrics.
- Tune fresh linear hyperparameters once on unedited decoder-fit data, freeze them for all edits, and repeat the locked MLP seeds.
- Persist per-example logits, probabilities, predictions, labels, group IDs, and edit IDs.

**Green test:** Run the focused test and validate exact key coverage for all 65 edits.

## Task 6: Block-aware analysis, figures, and manuscript patching

**Files:**

- Create: `src/reviewer_revision/analysis.py`
- Create: `src/reviewer_revision/figures.py`
- Create: `src/reviewer_revision/paper.py`
- Test: `tests/reviewer_revision/test_analysis.py`
- Test: `tests/reviewer_revision/test_figures_paper.py`

**Red test:** Require deterministic 10,000-draw hierarchical bootstrap, exact paired sign-flip p-values over 12 model-task blocks, refusal of incomplete key sets, vector and PNG figures generated without importing model code, macro values identical to validated summaries, and patching limited to named manuscript markers.

Run: `.venv\Scripts\python.exe -m pytest -q tests/reviewer_revision/test_analysis.py tests/reviewer_revision/test_figures_paper.py`

Expected red state: missing analysis/figure/paper modules.

**Implementation:**

- Materialize Parquet/CSV with stable schema and ordering.
- Aggregate within model-task blocks before primary inference; retain row-level descriptive summaries.
- Implement seeded hierarchical bootstrap and exact two-sided sign-flip tests.
- Render `fig_circularity_expanded`, `fig_epsilon_sweep`, and `fig_orientation_redecodability` as PDF plus 300-DPI PNG using the Okabe-Ito palette and no internal title.
- Generate TeX macros from summary JSON and patch only locked marker regions after all completeness gates pass.

**Green test:** Run focused tests, inspect generated image dimensions, and compare macro parses to summary values.

## Task 7: CLI and end-to-end dry run

**Files:**

- Create: `scripts/run_reviewer_revision.py`
- Create: `tests/reviewer_revision/test_cli.py`
- Modify: `README.md`

**Red test:** Exercise all required subcommands (`preflight`, `benchmark`, `reproduce-baseline`, `matched-split`, `epsilon-sweep`, `construct-check`, `analyze`, `figures`, `patch-paper`, `all`) with `--dry-run`, `--resume`, output-root, device, and log-level options. Require nonzero exits on validation failures and require `all` to stop before paper mutation when a prior gate fails.

Run: `.venv\Scripts\python.exe -m pytest -q tests/reviewer_revision/test_cli.py`

Expected red state: CLI absent.

**Implementation:**

- Route every subcommand through the validated config and run context.
- Mirror structured logging to console and the immutable run log.
- Resolve `auto` to MPS when available and otherwise CPU, while recording CUDA availability without selecting it.
- Make dry-run enumerate exact units and resource estimates without training.
- Make resume key-based and idempotent.

**Green test:** Run the CLI tests and a real preflight in a temporary output root.

## Task 8: Execute, audit, render, and commit

**Files:**

- Generate: `results/reviewer_revision_2026_08/<run-id>/**`
- Generate: `figures/fig_circularity_expanded.{pdf,png}`
- Generate: `figures/fig_epsilon_sweep.{pdf,png}`
- Generate: `figures/fig_orientation_redecodability.{pdf,png}`
- Generate: `manuscript_numbers.tex`
- Modify after gates: `main_revised.tex`

**Execution:**

1. Run the full untouched suite in the repository venv.
2. Run `preflight`, `benchmark`, and `reproduce-baseline`; inspect their manifests and select the locked full/fallback grid only by the benchmark thresholds.
3. Run matched/split, epsilon, and construct commands with resume enabled.
4. Run analysis and independently verify expected versus observed exact keys.
5. Generate figures and inspect rendered PNGs.
6. Patch the paper, compile with `latexmk`, render all pages, and inspect layout plus anonymous metadata.
7. Run the complete suite again, `git diff --check`, schema checks, input/output hashes, and manuscript-number consistency checks.

**Required commits:**

- `revision: add validated reviewer experiment driver`
- `revision: record expanded evaluator and attack sweep results`
- `paper: center evaluator coupling and add construct checks`

Do not stage or alter the user's pre-existing deletions of the legacy AAAI style and checklist files.
