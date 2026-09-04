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

### Data

- SVA: place numpred.{train,val,test} in data/.
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
