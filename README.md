# Probe Reliability through Geometric and Spectral Diagnostics

## Setup

### Windows

```
python -m venv .venv
.venv\Scripts\activate
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Lambda Cloud

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### HuggingFace gating

```
huggingface-cli login
```

Accept licenses on:
- https://huggingface.co/meta-llama/Llama-3.2-3B
- https://huggingface.co/google/gemma-2-2b

## Reviewer revision pipeline

Run the locked reviewer experiment from the repository root with the checked-in
Windows virtual environment. The required scientific protocol fits probes,
applies edits, and computes statistics on CPU; accelerator use is limited to
transformer representation extraction and the permitted candidate-MLP/Jacobian
work. The reviewer runner consumes and preflight-validates existing caches;
`--device auto` selects MPS for those accelerator-eligible operations on Apple
Silicon and otherwise records a CPU fallback. It never moves the locked small
scientific probes, edits, fresh decoders, or statistics off CPU.

```powershell
# First invocation: creates a new immutable timestamped run directory.
.\.venv\Scripts\python.exe -m scripts.run_reviewer_revision all `
  --config .\revision_experiment_spec.yaml `
  --output-root .\results\reviewer_revision_2026_08 `
  --device auto

# Resume after an interruption: reopens the newest compatible unlocked run.
.\.venv\Scripts\python.exe -m scripts.run_reviewer_revision all `
  --config .\revision_experiment_spec.yaml `
  --output-root .\results\reviewer_revision_2026_08 `
  --device auto `
  --resume
```

Each first invocation creates
`results/reviewer_revision_2026_08/<UTC timestamp>-<commit>/`. Existing run
directories and completed shards are never overwritten. `--resume` requires
the same configuration hash and Git commit, validates completed artifacts and
their hashes, and skips only gates that remain complete.

The fail-closed gate order is: `preflight` -> `benchmark` ->
`reproduce-baseline` -> `matched-split` -> `epsilon-sweep` ->
`construct-check` -> `analyze` -> `figures` -> `patch-paper`. A failed gate
stops all downstream work. The run directory contains the manifest and log,
gate reports, CSV/Parquet row tables and summaries, analysis/macros/validation
artifacts, PDF/PNG figures, and the compiled manuscript PDF specified by
`revision_experiment_spec.yaml`.

An `all` invocation intentionally finishes with
`pending_visual_inspection`, even when it exits successfully. The handoff is
complete only after every rendered manuscript page and generated figure has
been visually inspected and that review is recorded as `visually_inspected`;
an exit code or source-level test result alone is not the completion gate.

### Reviewer caveat-hardening extension

The extension adds the locked denominator-floor sensitivity analysis and the
prospective 11-cell construct panel. Run it from the repository root after the
base reviewer pipeline has completed and its run has passed finalization.

```powershell
# Check the exact workload without computing endpoints.
.\.venv\Scripts\python.exe .\scripts\run_reviewer_caveat_extension.py all `
  --config .\revision_caveat_extension_spec.yaml `
  --base-config .\revision_experiment_spec.yaml `
  --dry-run --device cuda

# Registered submission run. The extension spec names the immutable base run.
.\.venv\Scripts\python.exe .\scripts\run_reviewer_caveat_extension.py all `
  --config .\revision_caveat_extension_spec.yaml `
  --base-config .\revision_experiment_spec.yaml `
  --output-root .\results\reviewer_caveat_extension_2026_08 `
  --device cuda

# Resume the newest compatible run after interruption.
.\.venv\Scripts\python.exe .\scripts\run_reviewer_caveat_extension.py all `
  --config .\revision_caveat_extension_spec.yaml `
  --base-config .\revision_experiment_spec.yaml `
  --output-root .\results\reviewer_caveat_extension_2026_08 `
  --device cuda --resume
```

For a clean, code-only reproduction, first regenerate and finalize the base
reviewer run with the command above, then supply that timestamped directory
explicitly. The opt-in flag preserves the registered default while allowing a
new base run only after the same schema, configuration hash, finalization, and
artifact-integrity checks pass.

```powershell
.\.venv\Scripts\python.exe .\scripts\run_reviewer_caveat_extension.py all `
  --config .\revision_caveat_extension_spec.yaml `
  --base-config .\revision_experiment_spec.yaml `
  --base-run .\results\reviewer_revision_2026_08\<run-id> `
  --allow-reproduced-base `
  --output-root .\results\reviewer_caveat_extension_2026_08 `
  --device cuda
```

Preflight validates the base artifacts, all model/task/layer cache pins, and a
storage budget equal to the projected construct artifacts plus an 8 GiB free
reserve. The stages are `preflight`, `robustness`, `construct-panel`, `analyze`,
`package-artifacts`, `figures`, and `patch-paper`. A successful run produces
hash-verified row tables, inference summaries, environment locks, figures, and
the compiled manuscript beneath one immutable timestamped output directory.
Generated rows, checkpoints, caches, figures, and manuscript builds are local
artifacts and are deliberately not committed to Git. Render and inspect every
page of the final PDF and each generated figure before submission.

### Data

- SVA: place `numpred.{train,val,test}` from the MIT-licensed
  [Linzen agreement repository](https://github.com/TalLinzen/rnn_agreement)
  in `data/`.
- Gender agreement: data/gender.tsv (TAB-separated:
  sentence  FEM|MASC  FEM_SKEW|MASC_SKEW), REQUIRED. The checked-in file
  is the regenerated v2 dataset: 188 examples, full-sentence style,
  occupation-referent Winogender templates, BLS skew thresholds
  (%female ≥ 60 → FEM_SKEW, ≤ 40 → MASC_SKEW, NEUTRAL dropped); full
  provenance in data/gender.provenance.json. To regenerate, run
  `python -m scripts.generate_gender_data` (see that script's docstring
  for Winogender source options). The v1 synthetic fallback is deleted —
  it silently produced sentences truncated before the pronoun, making
  the task unlearnable — and the v2 loader rejects any such artifact.
- SST-2: optional data/sst2.tsv (TAB: sentence  0|1). If absent,
  loads from HuggingFace stanfordnlp/sst2.

## Verification

```
# ~3 min on 5070
python -m scripts.smoke_test

# ~5 min full pipeline check
python -m scripts.run_model --config configs/tiny.yaml

# ~10 min benchmark check (k=2, single layer)
python -m scripts.run_benchmark --config configs/tiny.yaml --task sva --k 2 --layers 6
```

If those pass, the system is healthy.

## Post-registration robustness study

The schema-v3 runner adds fresh post-edit decoders, orientation-robust metrics,
matched baselines, grouped five-way splits, Hessian subspace diagnostics, and
perturbation sweeps. It writes to a separate results directory and does not
alter the preregistered schema-v2 outcome.

```bash
# One-model smoke test
python -m scripts.ws9_robustness_sweep \
  --models pythia --tasks sva --smoke --analyze

# Full synthetic control grid, including the registered correlated condition
python -m scripts.ws9_synthetic_controls

# Full real-data robustness sweep
python -m scripts.ws9_robustness_sweep \
  --save-predictions --gender-pre-pronoun --analyze
```

See [docs/ROBUSTNESS_V3.md](docs/ROBUSTNESS_V3.md) for the estimands, split
contract, output schema, and interpretation rules.

## Production: Per-Model Pipeline (Original Failure-Mode Analysis)

```
python -m scripts.run_model --config configs/pythia.yaml
python -m scripts.run_model --config configs/bert.yaml
python -m scripts.run_model --config configs/gpt2.yaml
python -m scripts.run_model --config configs/qwen.yaml
python -m scripts.run_model --config configs/llama.yaml
python -m scripts.run_model --config configs/gemma.yaml

python -m scripts.aggregate
```

Writes results/<model>/results.json and results/aggregate/main_table.csv.

For each (model, task) combination, train K=20 seeded probes per
(architecture, layer) cell:

```
# Linux/macOS:
for cfg in configs/pythia.yaml configs/bert.yaml configs/gpt2.yaml \
           configs/qwen.yaml configs/llama.yaml configs/gemma.yaml; do
    for task in sva gender sst2; do
        python -m scripts.run_benchmark --config "$cfg" --task "$task" --k 20
    done
done
```

```
# Windows PowerShell:
foreach ($cfg in @("pythia","bert","gpt2","qwen","llama","gemma")) {
    foreach ($task in @("sva","gender","sst2")) {
        python -m scripts.run_benchmark --config "configs/$cfg.yaml" --task $task --k 20
    }
}
```

Each run writes results/benchmark_v2/<model>_<task>.jsonl (schema v2),
one JSON record per probe (5,400 records total at full scale), plus a
.manifest.json with config, git commit, data hashes, and the
learnability-gate record for each (re)launch. Runs are resume-safe:
completed (layer, arch, seed) cells found in the output file are skipped.
Before launching, a learnability gate (src/gates.py) checks that Zc and
Ze are linearly decodable at the middle probed layer and that a
shuffled-label control sits at chance; the run refuses to start otherwise.
The tainted v1 records remain untouched in results/benchmark/ and are
analyzed only by the locked v1 evaluator (scripts/predictor_eval.py);
analysis scripts take an explicit --input-dir and refuse to mix schema
versions.

## Methodology Highlights

### Extraction
hidden_states[layer] at last non-padding input token.
add_special_tokens=False to prevent BERT's [SEP] from corrupting
extraction. Validated on every run.

### Predictor
For probe parameters w (flattened across all trainable params in
model.parameters() order) and Hessian eigenvectors v_i (flattened
in the same order):

    A = mean( |cos(w, v_i)| for i in top_20 )

Computed using PyHessian's stochastic power iteration on the probe's
training-loss Hessian.

### Reliability
Per-probe R = max_method harmonic_mean(C, S) across the five
intervention methods, where:
- C = clip((acc_zc_pre - acc_zc_post) / (acc_zc_pre - 0.5), 0, 1)
- S = clip((acc_ze_post - 0.5) / (acc_ze_pre - 0.5), 0, 1)

Validation probes (linear, trained on un-intervened representations) are
applied unchanged before and after intervention to prevent the
intervention itself from leaking discriminative signal.

### Tasks
- SVA (Linzen 2016): Zc = verb number, Ze = last-noun number
- Gender: Zc = pronoun gender, Ze = profession's stereotypical gender
- SST-2: Zc = sentiment, Ze = sentence length bucket
