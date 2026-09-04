# WS0–WS3: experimenter decisions pending sign-off

> **SUPERSEDED by `PREREGISTRATION_v2.md`.** Every decision below has since
> been locked there; this doc is kept as a historical record and some values
> in it are stale. In particular: `mka.lambda_reg` is locked at **0.5**
> (§3.3, and all `configs/*.yaml` carry 0.5), not the provisional 0.1 in
> item 1; and `data/gender.tsv` has been **regenerated** (188 examples,
> full-sentence style, commit `1f68c21` — see `data/gender.provenance.json`
> and §3.4), so item 4's ACTION about the broken v1 artifact is done.

The WS0–WS3 code landed with provisional defaults everywhere a value is an
experimenter call. Each item below must be confirmed or overridden, then
locked in `PREREGISTRATION_v2.md` **before** the v2 benchmark launches.
Nothing in this list blocks code review; all of it blocks the re-run.

## 1. MKA regularization weight λ (WS1)

- Where: `configs/*.yaml` → `mka.lambda_reg` (currently 0.1, the v1 value).
- Evidence: `python -m scripts.mka_lambda_sweep --synthetic` (committed run:
  `results/mka_sweep/synthetic.jsonl`). On synthetic data, hidden-manifold
  alignment rises monotonically with λ (0.231 at λ=0 → 0.299 at λ=3),
  parameters diverge from the seed-matched MLP from λ=0.01 up, and test
  accuracy is flat until a small dip at λ≥1. λ in [0.3, 1.0] is where the
  regularizer visibly shapes training without cost on synthetic data.
- Needed: run the sweep on real representations once a model is available
  (`--config configs/tiny.yaml --task sva`) and pick λ personally.

## 2. Kernel construction detail (WS1, informational)

- Hidden-side kernel is a soft RBF whose bandwidth is the median squared
  distance to the k-th nearest neighbor (same k as the hard input kNN
  kernel), detached. Plain median-of-all-pairs went nearly flat in higher
  dimensions (identity-map alignment 0.13 vs 0.40+ with the k-th-neighbor
  bandwidth). Input kernel stays hard binary kNN.
- Also note: `mka_score` had a latent denominator bug (cross-term
  `mean(K)·mean(L)` in both variance terms) that was harmless for v1's
  binary/binary case but explodes (~1e6) for kernels with unequal means.
  Fixed to the standard centered-cosine form; identical behavior when
  kernel means match, i.e. for all v1 records.

## 3. Decodability floor (WS3)

- Where: `src/metrics.py` → `DECODABILITY_FLOOR = 0.55`; runner override
  `--floor`.
- Below the floor, C/S/R are null (JSON `null`), never clipped; legacy
  clipped values ride along as `*_clipped`, raw unclipped ratios as
  `*_raw`, plus both room denominators. Needed: confirm 0.55 or set your
  own, and lock it.

## 4. Gender task design (WS2)

- Style: `full-sentence` is the generator default (pronoun included, whole
  sentence kept; last-token extraction then tests persistence of gender to
  sentence end). `--style post-pronoun` exists as an explicit ablation.
  Needed: confirm full-sentence as the registered choice.
- NEUTRAL occupations: dropped by the generator; the v2 loader hard-rejects
  NEUTRAL labels. Needed: confirm (or write down a reason to keep them).
- Skew thresholds: BLS %female ≥ 60 → FEM_SKEW, ≤ 40 → MASC_SKEW,
  in between → dropped as NEUTRAL (`--fem-threshold` / `--masc-threshold`,
  `--stat-source bls|bergsma`). Needed: confirm thresholds and source.
- Referent: only occupation-referent templates by default (`--referent`).
- ACTION: `data/gender.tsv` in the repo is still the broken v1 artifact
  (kept for the v1 record; the v2 loader refuses it with a pointed error).
  Regenerate it on a machine with network access:
  `python -m scripts.generate_gender_data --download`
  (this session's sandbox could not reach the Winogender sources, so the
  regenerated file is not part of this change).

## 5. Learnability-gate floors (WS2)

- Where: `src/tasks.py` → `Task.zc_gate_floor = 0.60`,
  `Task.ze_gate_floor = 0.55` (per-task overrides go on the task classes).
- Gate mechanics: linear probe on the middle probed layer, 50/50 split,
  standardized features; plus a shuffled-label control that must sit within
  chance ± max(0.05, 4·binomial-sd). `run_benchmark` refuses to launch on
  failure (exit 2) and records the gate result in the manifest.
- Needed: registered floors per task.

## 6. v1 evaluator boundary (WS0.2, informational)

- `scripts/predictor_eval.py` gained an `--input-dir` argument and a guard
  that refuses schema-v2 rows. Its thresholds and statistics are untouched;
  it reproduces the committed v1 `PREREG_OUTCOME.json` byte-for-byte on the
  v1 records. The v2 evaluator (`prereg_v2_eval.py`, locked thresholds) is
  WS6 and does not exist yet.
