# Reviewer Revision Pipeline Design

## Objective

Implement the locked reviewer revision in `revision_experiment_spec.yaml`: reproduce the archived 12-cell control, expand matched-versus-split evaluation across depth, sweep FGSM/PGD budgets, run the prespecified Qwen/SST-2 construct check, generate auditable artifacts and figures, then patch and compile `main_revised.tex` only after every scientific gate passes.

## Architecture

The revision is an additive package under `src/reviewer_revision/`, driven by `scripts/run_reviewer_revision.py`. It reuses the repository's probe, attack, projection, candidate-direction, task, and representation-cache implementations. It does not modify archived schema-v2 or schema-v3 rows.

The package is split by responsibility:

- `config.py` validates the locked YAML and expands exact full/fallback/epsilon/construct keys.
- `artifacts.py` creates immutable run directories, hashes inputs, acquires a lock, writes validated shards atomically, and resumes from exact keys.
- `data.py` reconstructs archived Phase-2 folds, selects caches by verified data hash instead of glob order, validates tensors/provenance, and emits exact example/group manifests.
- `scoring.py` owns status-bearing C/S/H scores plus calibration-frozen orientation metrics.
- `training.py` owns deterministic probe/checkpoint training and fresh-decoder hyperparameter selection with inner group-disjoint validation.
- `experiments.py` runs matched/split pairs, the epsilon sweep, benchmarks, and the one-cell construct check while preserving the exact edited tensor for both scoring conditions.
- `analysis.py` checks complete key sets and computes block-aware summaries, exact sign-flip tests, and hierarchical bootstrap intervals from row artifacts.
- `figures.py` renders only saved rows to vector PDF and 300-DPI PNG.
- `paper.py` generates macros from validated summaries and replaces only marked manuscript regions.

## Data Flow

`preflight` validates companion hashes, environment, repository tests, cache provenance, finite/variance conditions, extraction position, and reconstructed split hashes. `benchmark` measures one pair in the two locked cells and selects the 60- or 36-cell grid. `reproduce-baseline` checks the archived JSON separately from the new group-aware estimand. Experiment commands write one atomic shard per exact unit, then materialize Parquet and CSV without duplicate keys. `analyze` recomputes every summary from row files and refuses incomplete or unexplained units. `figures` and `patch-paper` consume validated summaries only. `all` executes that order and stops at the first hard gate.

## Scientific Invariants

- One detached edited tensor and one SHA-256 edit hash are shared by matched and split rows.
- All requested units produce either an `ok` row or an explicit failure row.
- Attacker, split evaluator, scoring, direction, decoder-fit, calibration, and final-test groups are disjoint where required.
- The same model-task split IDs are reused across layers.
- Epsilon zero is a no-op and every perturbation stays inside its L-infinity ball.
- Orientation is chosen on calibration examples and frozen before final-test scoring.
- Fresh-decoder hyperparameters are selected once on unedited decoder-fit data and reused for every edit.
- Primary inference clusters the selected layers inside 12 model-task blocks.
- No manuscript mutation occurs before validated experiment and analysis completion.

## Failure and Resume Behavior

Every command returns nonzero on failed validation. Recoverable units are written to temporary shard paths, validated, and atomically renamed. Resume derives completed units from validated shard keys, never from row counts. A run lock prevents concurrent writers. Hard gates write structured blocker information to the run manifest and prevent downstream commands.

## Testing

Unit tests cover scoring, calibration, attacks, projections, padding, split/cache hashes, row keys, and bootstrap determinism. Integration tests cover shared edit hashes, six-way group disjointness, archived aggregate reproduction, atomic interruption/resume equivalence, summary-to-macro generation, and figure generation without model loading. The untouched full test suite is rerun after focused tests.

## Approved Inputs

The user supplied and explicitly approved the one-shot handoff, `main_revised.tex`, `references.bib`, and `revision_experiment_spec.yaml`, and instructed the agent to continue until completion. This document records that design without changing its locked scientific choices.
