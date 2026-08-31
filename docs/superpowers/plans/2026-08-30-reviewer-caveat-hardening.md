# Reviewer Caveat Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add preregistered multi-cell construct replication, denominator-floor robustness, portable reproduction metadata, a workshop-compliant paper update, and a code-only remote branch.

**Architecture:** Preserve the completed registered run as immutable input. Add a separately hashed extension configuration and focused extension modules that reuse the existing construct worker after parameterizing cell identity and artifact namespaces. Generate claims, figures, and manuscript macros only from validated row artifacts, then reconstruct the unpushed branch in an isolated worktree so only code and reproduction metadata reach GitHub.

**Tech Stack:** Python 3.13, PyTorch, pandas, NumPy, SciPy, scikit-learn, PyArrow, Matplotlib, pytest, Ruff, YAML, LaTeX/latexmk, Poppler, Git.

---

## File Map

- Create `revision_caveat_extension_spec.yaml` for prospective cells, thresholds, inference, storage gates, and outputs.
- Create `src/reviewer_revision/extension_config.py` for immutable extension-spec validation.
- Create `src/reviewer_revision/extension_analysis.py` for floor robustness and confirmatory construct inference.
- Create `src/reviewer_revision/extension_runner.py` and `scripts/run_reviewer_caveat_extension.py` for resumable orchestration.
- Create `src/reviewer_revision/portability.py` for environment locks and tiered content manifests.
- Modify `src/reviewer_revision/runner.py` to expose a parameterized construct-cell worker while preserving the original one-cell entry point.
- Modify `src/reviewer_revision/analysis.py`, `figures.py`, `paper.py`, and `main_revised.tex` for validated extension outputs.
- Add focused tests under `tests/reviewer_revision/` for every new behavior.

### Task 1: Lock and validate the extension specification

**Files:**
- Create: `revision_caveat_extension_spec.yaml`
- Create: `src/reviewer_revision/extension_config.py`
- Create: `tests/reviewer_revision/test_extension_config.py`

- [ ] **Step 1: Write failing exact-cell and pilot-exclusion tests**

```python
from pathlib import Path

import pytest
import yaml

from src.reviewer_revision.extension_config import ConstructCell, ExtensionConfigError, load_extension_config


def _valid_extension_payload():
    return yaml.safe_load(Path("revision_caveat_extension_spec.yaml").read_text(encoding="utf-8"))


def _write_yaml(tmp_path, payload):
    path = tmp_path / "extension.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_extension_config_locks_eleven_untouched_cells():
    config = load_extension_config("revision_caveat_extension_spec.yaml")
    assert config.pilot == ConstructCell("qwen", "Qwen/Qwen2.5-1.5B", "sst2", 14)
    assert len(config.confirmatory_cells) == 11
    assert config.pilot not in config.confirmatory_cells
    assert len(config.all_cells) == 12
    assert config.recovery_thresholds == {
        "accuracy": 0.55,
        "target_recovery_ratio": 0.50,
        "control_retention_ratio": 0.80,
    }


def test_extension_config_rejects_pilot_in_confirmatory_cells(tmp_path):
    payload = _valid_extension_payload()
    payload["construct_panel"]["confirmatory_cells"].append(payload["construct_panel"]["pilot"])
    path = _write_yaml(tmp_path, payload)
    with pytest.raises(ExtensionConfigError, match="pilot"):
        load_extension_config(path)
```

- [ ] **Step 2: Verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/reviewer_revision/test_extension_config.py -q`

Expected: collection fails because `extension_config` does not exist.

- [ ] **Step 3: Add the locked YAML and minimal configuration API**

```python
@dataclass(frozen=True, order=True)
class ConstructCell:
    model_key: str
    model_id: str
    task: str
    layer: int

    @property
    def slug(self) -> str:
        return f"{self.model_key}-{self.task}-l{self.layer}"


@dataclass(frozen=True)
class ExtensionConfig:
    source_path: Path
    config_hash: str
    base_run: Path
    pilot: ConstructCell
    confirmatory_cells: tuple[ConstructCell, ...]
    recovery_thresholds: Mapping[str, float]
    bootstrap_draws: int
    bootstrap_seed: int
    disk_reserve_gib: float

    @property
    def all_cells(self) -> tuple[ConstructCell, ...]:
        return (self.pilot, *self.confirmatory_cells)

    def validate(self) -> None:
        if self.pilot in self.confirmatory_cells:
            raise ExtensionConfigError("pilot cell must not appear in confirmatory cells")
        if len(set(self.confirmatory_cells)) != 11:
            raise ExtensionConfigError("exactly 11 unique confirmatory cells are required")
        expected = {"accuracy": 0.55, "target_recovery_ratio": 0.50, "control_retention_ratio": 0.80}
        if dict(self.recovery_thresholds) != expected:
            raise ExtensionConfigError("recovery thresholds differ from the locked design")
```

The YAML explicitly enumerates the pilot and 11 confirmatory cells, 10,000 bootstrap draws, Holm-adjusted cell p-values, an 8-GiB reserve, and immutable root `results/reviewer_caveat_extension_2026_08`.

- [ ] **Step 4: Verify GREEN and original-config compatibility**

Run: `.venv\Scripts\python.exe -m pytest tests/reviewer_revision/test_extension_config.py tests/reviewer_revision/test_config.py -q`

Expected: all tests pass and the original locked config remains unchanged.

- [ ] **Step 5: Commit the registration before computing endpoints**

```powershell
git add -- revision_caveat_extension_spec.yaml src/reviewer_revision/extension_config.py tests/reviewer_revision/test_extension_config.py
git commit -m "revision: preregister caveat hardening extension"
```

#### Prospective amendment recorded 2026-08-31 before confirmatory endpoints

Implementation review found inference conventions that the original YAML did not spell out. Before computing or inspecting any endpoint in the 11 confirmatory cells, amend and re-hash the registration to lock all of the following:

- `alpha = 0.05`, NumPy `PCG64`, and NumPy quantiles with `method="linear"`;
- endpoint p-value `(1 + count(theta_star <= threshold)) / (B + 1)` for each one-sided lower-tail test;
- within-cell intersection-union p-value `max(p_accuracy, p_target_recovery, p_control_retention)`;
- Holm adjustment across exactly 11 confirmatory cell p-values;
- final pass requires all three point thresholds and Holm-adjusted cell p-value at most `0.05`, with the point and inferential booleans emitted separately;
- a nonestimable confirmatory cell stays in the 11-cell family with internal p-value `1` and label `nonestimable`;
- the pilot is descriptive only and receives no confirmatory p-value or decision; and
- all one-sided 95% lower bounds are marginal, not Holm-adjusted and not simultaneous.

Do not run confirmatory endpoint analysis before the amended YAML, config validator, tests, design, and this plan are committed.

### Task 2: Add full-case and floor-robustness analyses

**Files:**
- Create: `src/reviewer_revision/extension_analysis.py`
- Create: `tests/reviewer_revision/test_extension_analysis.py`
- Modify: `src/reviewer_revision/analysis.py`

- [ ] **Step 1: Write failing raw-drop, floor-curve, and bound tests**

```python
import pandas as pd
import pytest

from src.reviewer_revision.extension_analysis import summarize_floor_robustness


def _matched_split_fixture_rows_with_one_floor_null():
    rows = []
    values = {
        ("bert", "sva", 6, 0): {"matched": (0.90, 0.40), "split": (0.80, 0.60)},
        ("gpt2", "sst2", 6, 0): {"matched": (0.54, 0.34), "split": (0.54, 0.44)},
    }
    for (model, task, layer, pair_seed), conditions in values.items():
        for condition, (pre, post) in conditions.items():
            rows.append({"model_key": model, "task": task, "layer": layer, "pair_seed": pair_seed, "method": "alterrep", "condition": condition, "target_acc_pre": pre, "target_acc_post": post})
    return rows


def test_floor_robustness_includes_every_raw_drop_pair():
    rows = pd.DataFrame(_matched_split_fixture_rows_with_one_floor_null())
    summary = summarize_floor_robustness(rows, draws=200, seed=7)
    raw = summary["full_case_raw_drop"]
    assert raw["pairs"] == 2
    assert raw["cells"] == 2
    assert raw["matched_mean"] == pytest.approx(0.35)
    assert raw["split_mean"] == pytest.approx(0.15)
    assert raw["gap"] == pytest.approx(0.20)
    assert raw["post_hoc_sensitivity"] is True


def test_partial_identification_contains_available_case_gap():
    rows = pd.DataFrame(_matched_split_fixture_rows_with_one_floor_null())
    bounds = summarize_floor_robustness(rows, draws=200, seed=7)["partial_identification"]
    assert bounds["missing_pair_gap_domain"] == [-1.0, 1.0]
    assert bounds["lower"] <= bounds["available_case_gap"] <= bounds["upper"]
```

- [ ] **Step 2: Verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/reviewer_revision/test_extension_analysis.py -q`

Expected: collection fails because `summarize_floor_robustness` is absent.

- [ ] **Step 3: Implement hierarchy-preserving sensitivities**

```python
FLOOR_GRID = (0.5000000001, 0.525, 0.55, 0.575, 0.60)


def raw_target_drop(frame: pd.DataFrame) -> pd.Series:
    return frame["target_acc_pre"].astype(float) - frame["target_acc_post"].astype(float)


def target_damage_at_floor(frame: pd.DataFrame, floor: float) -> pd.Series:
    pre = frame["target_acc_pre"].astype(float)
    damage = (pre - frame["target_acc_post"].astype(float)) / (pre - 0.5)
    return damage.clip(0.0, 1.0).where(pre >= floor)


def summarize_floor_robustness(rows: pd.DataFrame, *, draws: int, seed: int) -> dict[str, Any]:
    alterrep = _validate_alterrep_pair_rows(rows)
    alterrep["raw_drop"] = raw_target_drop(alterrep)
    raw = _paired_hierarchical_contrast(alterrep, value="raw_drop", draws=draws, seed=seed)
    curves = []
    for floor in FLOOR_GRID:
        work = alterrep.copy()
        work["floor_damage"] = target_damage_at_floor(work, floor)
        curves.append({"floor": floor, **_paired_hierarchical_contrast(work, value="floor_damage", draws=draws, seed=seed)})
    return {
        "schema_version": 1,
        "status": "ok",
        "label": "post_hoc_sensitivity",
        "full_case_raw_drop": {**raw, "post_hoc_sensitivity": True},
        "floor_curve": curves,
        "partial_identification": _partial_identification_bounds(alterrep, floor=0.55),
    }
```

Promote the original exact sign-flip and hierarchical-bootstrap primitives to documented internal helpers; do not duplicate randomization logic.

- [ ] **Step 4: Verify GREEN and the authoritative final-run values**

Run: `.venv\Scripts\python.exe -m pytest tests/reviewer_revision/test_extension_analysis.py tests/reviewer_revision/test_analysis.py -q`

Expected: all tests pass. The immutable final rows must regenerate raw-drop gap `0.24847267648882757`, interval `[0.2046648055280495, 0.2970943073101083]`, exact p `0.00048828125`, 300 pairs, 60 cells, and 12 positive blocks.

- [ ] **Step 5: Commit**

```powershell
git add -- src/reviewer_revision/analysis.py src/reviewer_revision/extension_analysis.py tests/reviewer_revision/test_extension_analysis.py
git commit -m "revision: add denominator-floor robustness analyses"
```

### Task 3: Parameterize and namespace one construct-cell worker

**Files:**
- Modify: `src/reviewer_revision/runner.py`
- Create: `tests/reviewer_revision/test_construct_multicell.py`
- Modify: `tests/reviewer_revision/test_runner.py`
- Modify: `tests/reviewer_revision/test_construct_check.py`

- [ ] **Step 1: Write failing cell-identity and path-collision tests**

```python
from src.reviewer_revision.extension_config import ConstructCell
from src.reviewer_revision.runner import construct_artifact_root, construct_row_identity


def test_construct_artifacts_are_namespaced_by_cell(tmp_path):
    bert = ConstructCell("bert", "google-bert/bert-base-uncased", "sva", 6)
    qwen = ConstructCell("qwen", "Qwen/Qwen2.5-1.5B", "sst2", 14)
    assert construct_artifact_root(tmp_path, bert) == tmp_path / "construct" / "cells" / "bert-sva-l6"
    assert construct_artifact_root(tmp_path, qwen) == tmp_path / "construct" / "cells" / "qwen-sst2-l14"


def test_construct_row_identity_contains_cell_key():
    cell = ConstructCell("bert", "google-bert/bert-base-uncased", "sva", 6)
    row = construct_row_identity(cell, edit_kind="dcand_crossfit", architecture="linear", seed=3)
    assert (row["model_key"], row["task"], row["layer"], row["candidate_seed"]) == ("bert", "sva", 6, 3)


def test_construct_group_rows_store_counts_not_precomputed_ratios():
    rows = _two_group_construct_rows()
    required = {
        "row_kind", "model_key", "model_id", "task", "layer",
        "evaluation_family", "decoder_seed", "label", "group_id",
        "n_examples", "correct_count", "status",
    }
    assert required <= set(rows.columns)
    assert not {"accuracy", "target_recovery_ratio", "control_retention_ratio"} & set(rows.columns)
    assert set(rows.loc[rows["row_kind"] == "post_edit", "edit_id"]) == set(EXPECTED_60_EDIT_IDS)
```

- [ ] **Step 2: Verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/reviewer_revision/test_construct_multicell.py -q`

Expected: imports fail because the APIs are absent.

- [ ] **Step 3: Add the parameterized worker and namespaced paths**

```python
def construct_artifact_root(run_dir: Path, cell: ConstructCell) -> Path:
    return run_dir / "construct" / "cells" / cell.slug


def construct_row_identity(cell: ConstructCell, *, edit_kind: str, architecture: str | None, seed: int) -> dict[str, Any]:
    return {
        "model_key": cell.model_key,
        "model_id": cell.model_id,
        "task": cell.task,
        "layer": cell.layer,
        "edit_kind": edit_kind,
        "architecture": architecture,
        "candidate_seed": seed,
    }


def run_construct_cell(context: RunContext, config: RevisionConfig, *, cell: ConstructCell, device: torch.device) -> dict[str, Any]:
    root = construct_artifact_root(context.run_dir, cell)
    root.mkdir(parents=True, exist_ok=True)
    reconstruction = _reconstruct_tasks(config)[cell.task]
    caches = _load_construct_caches(context, reconstruction=reconstruction, cell=cell)
    return _execute_construct_cell(context, config, cell=cell, caches=caches, artifact_root=root, device=device)
```

Make `run_construct_check` a compatibility wrapper for the original pilot. Add cell identity to shard keys and every split, direction, edit, baseline, hyperparameter, checkpoint, and reconstruction reference. Persist a normalized `construct_group_rows` table of sufficient statistics. Common columns are `row_kind`, full cell identity, `evaluation_family`, `decoder_seed`, target/control `label`, `group_id`, `n_examples`, `correct_count`, provenance hashes, and `status`. Baseline keys are `(cell, evaluation_family, decoder_seed, label, group_id)`. Post-edit rows add `edit_id`, `edit_object`, architecture, candidate seed, and edit hash; their keys are `(cell, edit_id, evaluation_family, decoder_seed, label, group_id)`. Persist the exact 60 candidate edit identities. Do not persist group-level recovery/retention ratios as inference inputs: Task 4 reconstructs example-weighted pooled accuracies and ratios from counts inside each draw.

- [ ] **Step 4: Verify GREEN and one-cell compatibility**

Run: `.venv\Scripts\python.exe -m pytest tests/reviewer_revision/test_construct_multicell.py tests/reviewer_revision/test_construct_check.py tests/reviewer_revision/test_runner.py -q`

Expected: all tests pass, including two-cell interruption/resume, exact group-row key/count validation, and original pilot fixtures.

- [ ] **Step 5: Commit**

```powershell
git add -- src/reviewer_revision/runner.py tests/reviewer_revision/test_construct_multicell.py tests/reviewer_revision/test_construct_check.py tests/reviewer_revision/test_runner.py
git commit -m "refactor: parameterize construct checks by cell"
```

### Task 4: Add prospective multi-cell inference

**Files:**
- Modify: `src/reviewer_revision/extension_analysis.py`
- Modify: `src/reviewer_revision/analysis.py`
- Create: `tests/reviewer_revision/test_construct_panel_analysis.py`

- [ ] **Step 1: Write failing classification, bootstrap, Holm, and pilot tests**

```python
from src.reviewer_revision.extension_analysis import classify_construct_panel


def _construct_panel_fixture():
    # Build the pilot plus all 11 locked confirmatory cells. Baseline and each
    # of the exact 60 edits contain target/control sufficient statistics for
    # two final-test groups. Ratios are deliberately absent.
    return make_full_locked_panel_group_rows(
        groups=("g1", "g2"),
        edit_ids=EXPECTED_60_EDIT_IDS,
        evaluation_family="fresh_linear",
        decoder_seed=0,
    )


def test_construct_panel_excludes_pilot_and_applies_three_thresholds():
    summary = classify_construct_panel(_construct_panel_fixture(), config=_test_extension_config(draws=500, seed=11))
    assert summary["confirmatory_cell_count"] == 11
    assert summary["pilot"]["confirmatory"] is False
    assert "confirmatory_p_value" not in summary["pilot"]
    assert "confirmatory_decision" not in summary["pilot"]
    passing = summary["cells"]["bert-sva-l6"]
    assert passing["passes_locked_point_thresholds"] is True
    assert passing["passes_holm_adjusted_inference"] is True
    assert passing["passes_locked_confirmatory_rule"] is True
    assert passing["lower_bound_scope"] == "marginal"
    assert passing["lower_bound_multiplicity_adjusted"] is False


def test_joint_task_bootstrap_is_deterministic_and_holm_adjusts_cell_p_values():
    config = _test_extension_config(draws=500, seed=11)
    first = classify_construct_panel(_construct_panel_fixture(), config=config)
    second = classify_construct_panel(_construct_panel_fixture(), config=config)
    assert first == second
    assert first["multiplicity"]["method"] == "holm_one_sided"
    assert first["multiplicity"]["family_size"] == 11
    assert first["bootstrap"]["bit_generator"] == "PCG64"
    assert first["bootstrap"]["quantile_method"] == "linear"


def test_nonestimable_cell_stays_in_fixed_family_with_p_one():
    rows = make_one_confirmatory_cell_nonestimable(_construct_panel_fixture())
    summary = classify_construct_panel(rows, config=_test_extension_config(draws=500, seed=11))
    cell = summary["cells"][NONESTIMABLE_CELL_SLUG]
    assert cell["status"] == "nonestimable"
    assert cell["internal_cell_p_value"] == 1.0
    assert summary["multiplicity"]["family_size"] == 11
```

- [ ] **Step 2: Verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/reviewer_revision/test_construct_panel_analysis.py -q`

Expected: import fails for `classify_construct_panel`.

- [ ] **Step 3: Implement cell-first aggregation and task-joint resampling**

```python
LOCKED_RECOVERY_THRESHOLDS = {"accuracy": 0.55, "target_recovery_ratio": 0.50, "control_retention_ratio": 0.80}


def passes_recovery_thresholds(metrics: Mapping[str, float]) -> bool:
    return all(float(metrics[name]) >= threshold for name, threshold in LOCKED_RECOVERY_THRESHOLDS.items())


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    adjusted, running, count = {}, 0.0, len(ordered)
    for index, key in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * float(p_values[key])))
        adjusted[key] = running
    return adjusted


def classify_construct_panel(rows: pd.DataFrame, *, config: ExtensionConfig) -> dict[str, Any]:
    validated = _validate_construct_group_rows(
        rows,
        expected_cells=config.all_cells,
        expected_edit_ids=LOCKED_60_EDIT_IDS,
    )
    points = _cell_median_endpoints_from_group_counts(validated, config=config)
    bootstrap = _joint_task_group_bootstrap(
        validated,
        cells=config.confirmatory_cells,
        draws=config.bootstrap_draws,
        seed=config.bootstrap_seed,
        bit_generator="PCG64",
    )
    raw_cell_p = {}
    for cell in config.confirmatory_cells:
        slug = cell.slug
        if bootstrap[slug]["status"] == "nonestimable":
            raw_cell_p[slug] = 1.0
            continue
        endpoint_p = {
            name: (1 + np.count_nonzero(bootstrap[slug][name] <= threshold))
            / (config.bootstrap_draws + 1)
            for name, threshold in LOCKED_RECOVERY_THRESHOLDS.items()
        }
        bootstrap[slug]["endpoint_p_values"] = endpoint_p
        raw_cell_p[slug] = max(endpoint_p.values())  # intersection-union test
    adjusted = holm_adjust(raw_cell_p)  # exactly 11 entries, including p=1
    cells = _assemble_cell_records(points, bootstrap, adjusted, config=config)
    return {
        "schema_version": 1,
        "status": "ok",
        "pilot": cells[config.pilot.slug],
        "confirmatory_cell_count": 11,
        "cells": cells,
        "multiplicity": {"method": "holm_one_sided", "family_size": 11},
        "lower_bounds": {
            "confidence_level": 0.95,
            "scope": "marginal",
            "multiplicity_adjusted": False,
            "simultaneous": False,
        },
    }
```

The bootstrap resamples final-test group IDs once per task/draw and applies the same draw multiplicities to every model sharing that task. Inside each draw it recomputes example-weighted pooled baseline and post-edit accuracies from `correct_count / n_examples`, recomputes unclipped ratios, and then takes the median across the fixed dependent set of 60 candidate edits. A target baseline at or below `0.5` is a conservative endpoint-tail failure. Marginal lower bounds are the NumPy linear-method 0.05 quantiles. Point-threshold, Holm-inference, and combined decisions are separate booleans. A scientific threshold miss is status `ok`, not an execution failure. It never treats 60 edits as independent.

- [ ] **Step 4: Verify GREEN and original-analysis stability**

Run: `.venv\Scripts\python.exe -m pytest tests/reviewer_revision/test_construct_panel_analysis.py tests/reviewer_revision/test_analysis.py -q`

Expected: all tests pass and the original one-cell fixture is unchanged.

- [ ] **Step 5: Commit**

```powershell
git add -- src/reviewer_revision/analysis.py src/reviewer_revision/extension_analysis.py tests/reviewer_revision/test_construct_panel_analysis.py tests/reviewer_revision/test_analysis.py
git commit -m "revision: add confirmatory construct panel inference"
```

### Task 5: Add environment and artifact portability manifests

**Files:**
- Create: `src/reviewer_revision/portability.py`
- Create: `tests/reviewer_revision/test_portability.py`
- Modify: `src/reviewer_revision/artifacts.py`

- [ ] **Step 1: Write failing sanitization and hash-verification tests**

```python
import pytest

from src.reviewer_revision.portability import build_artifact_manifest, build_environment_lock, verify_artifact_manifest


def test_environment_lock_rejects_direct_urls(monkeypatch):
    monkeypatch.setattr("src.reviewer_revision.portability.installed_distributions", lambda: [{"name": "private", "version": "1.0", "direct_url": "file:///C:/Users/name/private"}])
    with pytest.raises(ValueError, match="direct URL"):
        build_environment_lock(spec_hash="a" * 64, git_commit="b" * 40)


def test_tier_manifest_detects_hash_mismatch(tmp_path):
    artifact = tmp_path / "rows.json"
    artifact.write_text("{}\n", encoding="utf-8")
    manifest = build_artifact_manifest(tmp_path, {"analysis": [artifact]})
    verify_artifact_manifest(tmp_path, manifest)
    artifact.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_artifact_manifest(tmp_path, manifest)
```

- [ ] **Step 2: Verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/reviewer_revision/test_portability.py -q`

Expected: import fails because `portability` does not exist.

- [ ] **Step 3: Implement canonical records and verification**

```python
def _portable_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def build_artifact_manifest(root: Path, tiers: Mapping[str, Sequence[Path]]) -> dict[str, Any]:
    memberships: dict[str, set[str]] = {}
    for tier, paths in tiers.items():
        for path in paths:
            memberships.setdefault(_portable_relative(path, root), set()).add(tier)
    records = []
    for relative, member_tiers in sorted(memberships.items()):
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty artifact: {relative}")
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path), "tiers": sorted(member_tiers)})
    return {"schema_version": 1, "files": records, "aggregate_sha256": sha256_json(records)}


def verify_artifact_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    records = list(manifest["files"])
    if sha256_json(records) != manifest["aggregate_sha256"]:
        raise ValueError("artifact manifest aggregate hash mismatch")
    for record in records:
        path = root / str(record["path"])
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"artifact size mismatch: {record['path']}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"artifact hash mismatch: {record['path']}")
```

`build_environment_lock` sorts canonical distribution names and rejects direct URLs, editable/local paths, credentials, and absolute-path metadata. It emits JSON plus safe `name==version` text entries.

- [ ] **Step 4: Verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/reviewer_revision/test_portability.py tests/reviewer_revision/test_artifacts.py -q`

Expected: all tests pass, including path escape, empty file, duplicate path, altered hash, and stable aggregate cases.

- [ ] **Step 5: Commit**

```powershell
git add -- src/reviewer_revision/artifacts.py src/reviewer_revision/portability.py tests/reviewer_revision/test_portability.py tests/reviewer_revision/test_artifacts.py
git commit -m "feat: add portable reviewer artifact manifests"
```

### Task 6: Add the extension CLI and resumable orchestration

**Files:**
- Create: `src/reviewer_revision/extension_runner.py`
- Create: `scripts/run_reviewer_caveat_extension.py`
- Create: `tests/reviewer_revision/test_extension_runner.py`
- Modify: `tests/reviewer_revision/test_cli.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing CLI and disk-gate tests**

```python
import pytest

from scripts.run_reviewer_caveat_extension import COMMANDS
from src.reviewer_revision.extension_runner import require_construct_disk_capacity


def test_extension_cli_exposes_every_stage():
    assert COMMANDS == ("preflight", "robustness", "construct-panel", "analyze", "package-artifacts", "figures", "patch-paper", "all")


def test_construct_disk_gate_requires_projection_plus_reserve():
    require_construct_disk_capacity(free_bytes=21 * 2**30, projected_bytes=12 * 2**30, reserve_bytes=8 * 2**30)
    with pytest.raises(RuntimeError, match="disk gate"):
        require_construct_disk_capacity(free_bytes=19 * 2**30, projected_bytes=12 * 2**30, reserve_bytes=8 * 2**30)
```

- [ ] **Step 2: Verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/reviewer_revision/test_extension_runner.py tests/reviewer_revision/test_cli.py -q`

Expected: imports fail because the extension CLI and runner are absent.

- [ ] **Step 3: Implement stages, disk gate, and exact-key resume**

```python
STAGES = ("preflight", "robustness", "construct-panel", "analyze", "package-artifacts", "figures", "patch-paper")


def require_construct_disk_capacity(*, free_bytes: int, projected_bytes: int, reserve_bytes: int) -> None:
    required = projected_bytes + reserve_bytes
    if free_bytes < required:
        raise RuntimeError(f"construct disk gate failed: free={free_bytes}, required={required}")


def run_construct_panel(context: RunContext, extension: ExtensionConfig, base: RevisionConfig, *, device: torch.device) -> dict[str, Any]:
    completed = validated_completed_cells(context, extension.confirmatory_cells)
    for cell in extension.confirmatory_cells:
        if cell in completed:
            continue
        result = run_construct_cell(context, base, cell=cell, device=device)
        context.write_json_shard("construct-panel", (cell.model_key, cell.task, cell.layer), result)
    assert_exact_keys(
        ((cell.model_key, cell.task, cell.layer) for cell in extension.confirmatory_cells),
        observed_construct_cell_keys(context),
    )
    return consolidate_construct_panel(context, extension)
```

Preflight validates base-run hashes, registration commit, caches for every cell/tag, deterministic CPU routing, and projected storage plus reserve. `all` stops at the first failed stage and binds resume to exact config hash and Git commit.

- [ ] **Step 4: Document exact code-only reproduction commands**

```powershell
.venv\Scripts\python.exe -m scripts.run_reviewer_caveat_extension all `
  --config revision_caveat_extension_spec.yaml `
  --base-config revision_experiment_spec.yaml `
  --device cpu
```

README must state that Git contains code and metadata only; weights, caches, rows, checkpoints, tensors, figures, and PDFs regenerate locally and are not redistributed.

- [ ] **Step 5: Verify GREEN and dry-run counts**

Run: `.venv\Scripts\python.exe -m pytest tests/reviewer_revision/test_extension_runner.py tests/reviewer_revision/test_cli.py tests/reviewer_revision/test_runner.py -q`

Expected: all tests pass.

Run: `.venv\Scripts\python.exe -m scripts.run_reviewer_caveat_extension all --config revision_caveat_extension_spec.yaml --base-config revision_experiment_spec.yaml --dry-run --device cpu`

Expected: 11 confirmatory cells, the existing 4,290 compatibility rows plus validated group sufficient-statistic rows for every baseline and exact candidate edit, 10,000 bootstrap draws, and no model loading.

- [ ] **Step 6: Commit**

```powershell
git add -- README.md scripts/run_reviewer_caveat_extension.py src/reviewer_revision/extension_runner.py tests/reviewer_revision/test_cli.py tests/reviewer_revision/test_extension_runner.py
git commit -m "feat: add resumable caveat extension driver"
```

### Task 7: Generate extension figures and paper claims

**Files:**
- Modify: `src/reviewer_revision/figures.py`
- Modify: `src/reviewer_revision/paper.py`
- Modify: `tests/reviewer_revision/test_figures_paper.py`
- Modify: `main_revised.tex`
- Modify: `assets/reviewer_revision/main_revised_prepatch.tex`

- [ ] **Step 1: Write failing paper-claim and panel-figure tests**

```python
def _extension_summary_fixture(*, confirmatory_cells: int, robustness_label: str):
    return {
        "floor_robustness": {"label": robustness_label},
        "construct_panel": {
            "confirmatory_cell_count": confirmatory_cells,
            "pilot": {"confirmatory": confirmatory_cells == 12},
            "cells": {},
        },
    }


def test_extension_paper_requires_post_hoc_label_and_eleven_cells(tmp_path):
    summary = _extension_summary_fixture(confirmatory_cells=11, robustness_label="post_hoc_sensitivity")
    report = patch_manuscript_with_extension(PAPER_TEMPLATE, tmp_path / "paper.tex", summary)
    text = report.destination.read_text(encoding="utf-8")
    assert "post-hoc full-case sensitivity" in text
    assert "11 untouched middle-layer cells" in text
    assert "does not prove concept erasure" in text


def test_extension_paper_rejects_pilot_as_confirmatory(tmp_path):
    summary = _extension_summary_fixture(confirmatory_cells=12, robustness_label="post_hoc_sensitivity")
    with pytest.raises(ManuscriptValidationError, match="11 confirmatory"):
        patch_manuscript_with_extension(PAPER_TEMPLATE, tmp_path / "paper.tex", summary)
```

- [ ] **Step 2: Verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/reviewer_revision/test_figures_paper.py -q`

Expected: new tests fail because extension patching and plotting are absent.

- [ ] **Step 3: Add allowlisted regions and strict summary validation**

```python
def validate_extension_summary(summary: Mapping[str, Any]) -> None:
    robustness = _require_mapping(summary, "floor_robustness", "extension")
    if robustness.get("label") != "post_hoc_sensitivity":
        raise ManuscriptValidationError("floor robustness must remain post-hoc sensitivity")
    construct = _require_mapping(summary, "construct_panel", "extension")
    if construct.get("confirmatory_cell_count") != 11:
        raise ManuscriptValidationError("construct panel must contain exactly 11 confirmatory cells")
    if construct.get("pilot", {}).get("confirmatory") is not False:
        raise ManuscriptValidationError("construct pilot must remain non-confirmatory")
```

Generated prose reports pass or failure without a success-biased branch. The epsilon conclusion, rejected nonlinear gate, fixed-decoder limitation, and downstream-behavior limitation remain mandatory strings checked by tests.

- [ ] **Step 4: Add floor and construct-panel plots from saved outputs**

```python
def create_construct_panel_figure(summary_path: Path, output_dir: Path) -> FigureArtifacts:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cells = pd.DataFrame(summary["construct_panel"]["cells"].values())
    if len(cells.loc[cells["confirmatory"]]) != 11:
        raise FigureValidationError("construct panel requires 11 confirmatory cells")
    return _render_construct_panel(cells, output_dir, dpi=300)
```

Labels distinguish pilot, untouched cells, point thresholds, and marginal one-sided bounds. Captions state that the bounds are neither Holm-adjusted nor simultaneous. Save vector PDF plus 300-DPI PNG using the existing paper palette.

- [ ] **Step 5: Verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/reviewer_revision/test_figures_paper.py -q`

Expected: all tests pass, including original one-cell fixtures.

- [ ] **Step 6: Commit source-level paper support**

```powershell
git add -- main_revised.tex assets/reviewer_revision/main_revised_prepatch.tex src/reviewer_revision/figures.py src/reviewer_revision/paper.py tests/reviewer_revision/test_figures_paper.py
git commit -m "paper: add caveat extension reporting"
```

### Task 8: Run smoke validation and the full prospective panel

**Files:**
- Generate locally only: `results/reviewer_caveat_extension_2026_08/<run-id>/`

- [ ] **Step 1: Run code gates before scientific execution**

```powershell
.venv\Scripts\python.exe -m pytest tests/reviewer_revision -q
.venv\Scripts\python.exe -m pytest -q
ruff check scripts/run_reviewer_revision.py scripts/run_reviewer_caveat_extension.py src/reviewer_revision tests/reviewer_revision
git diff --check
```

Expected: zero failures and zero lint/whitespace errors. Existing third-party SWIG warnings may remain identified as pre-existing.

- [ ] **Step 2: Run one untouched-cell smoke execution**

```powershell
.venv\Scripts\python.exe -m scripts.run_reviewer_caveat_extension construct-panel `
  --config revision_caveat_extension_spec.yaml `
  --base-config revision_experiment_spec.yaml `
  --device cpu `
  --smoke-cell bert:sva:6
```

Expected: one cell, 65 compatibility edit records, 390 compatibility summary rows, complete baseline and exact-60-candidate group sufficient-statistic rows, valid hashes, no hard failures, and no base-run mutation.

- [ ] **Step 3: Run the full extension from a clean generating commit**

```powershell
.venv\Scripts\python.exe -m scripts.run_reviewer_caveat_extension all `
  --config revision_caveat_extension_spec.yaml `
  --base-config revision_experiment_spec.yaml `
  --device cpu
```

Expected: disk gate passes before loading models; robustness covers 300 pairs/60 cells; 11 cells produce 715 compatibility edit records, 4,290 compatibility summary rows, and complete normalized group sufficient-statistic rows or explicit fatal-unit records; analysis and manifests complete; paper ends pending visual inspection.

- [ ] **Step 4: Resume and validate every saved stage**

```powershell
.venv\Scripts\python.exe -m scripts.run_reviewer_caveat_extension all `
  --config revision_caveat_extension_spec.yaml `
  --base-config revision_experiment_spec.yaml `
  --device cpu `
  --resume
```

Expected: completed stages are hash-validated and skipped with no duplicate keys or changed outputs.

### Task 9: Finalize and visually inspect the workshop paper

**Files:**
- Generate locally only: extension figures, macros, rendered pages, compiled PDF, validation reports.

- [ ] **Step 1: Verify current workshop rules from the official CFP**

Open `https://interpretability4discovery.github.io/cfp.html` and record the current main-text limit, excluded sections, anonymity rule, and submission format in the validation report.

Expected: official rule evidence is linked and paraphrased without excessive quotation.

- [ ] **Step 2: Compile and run machine-verifiable gates**

```powershell
.venv\Scripts\python.exe -m scripts.run_reviewer_caveat_extension patch-paper `
  --config revision_caveat_extension_spec.yaml `
  --base-config revision_experiment_spec.yaml `
  --device cpu `
  --resume
```

Expected: latexmk exits zero; PDF author metadata is anonymous; the five-main-page workshop limit is satisfied with references starting afterward; no undefined citations/references; overfull boxes at most 3 pt; required PDF/300-DPI PNG figures exist.

- [ ] **Step 3: Inspect every page and generated figure**

Inspect full-resolution page images for title/anonymity, numbers, table ordering, captions, clipping, margins, legends, thresholds, pilot labeling, references, appendices, and the main-text boundary. Inspect each generated PNG at 300 DPI.

Expected: `visual_inspection_report.json` records every page and figure as passed or names a correction. Corrections trigger recompilation and complete reinspection.

- [ ] **Step 4: Run final numerical, test, and anonymity audit**

```powershell
.venv\Scripts\python.exe -m pytest -q
ruff check scripts/run_reviewer_revision.py scripts/run_reviewer_caveat_extension.py src/reviewer_revision tests/reviewer_revision
git diff --check
```

Scan source, extracted PDF text, reports, and intended commit paths for home paths, usernames, credentials, tokens, and non-anonymous metadata. Review citation-author and `aclanthology.org` matches manually.

Expected: all gates pass, extension manifest is `complete_visually_inspected`, and registered artifact hashes are unchanged.

### Task 10: Construct and push a code-only branch

**Files:**
- Preserve locally: result-bearing history and all untracked run directories.
- Push: source, tests, specs, docs, templates, and reproduction tooling only.

- [ ] **Step 1: Create a backup ref and isolated publication worktree**

```powershell
git branch local/reviewer-revision-with-results HEAD
git worktree add ..\probeReliabilityScript-code-only origin/main
```

Resolve and verify both worktree paths remain inside `C:\Users\antho\Documents\Projects\AI Research` before later cleanup.

- [ ] **Step 2: Rebuild implementation history without result commits**

```powershell
git switch -c agent/reviewer-revision-2026-08-code-only
git cherry-pick --no-commit 9b05a32
git restore --source=HEAD --staged --worktree -- aaai2027.bib aaai2027.bst aaai2027.sty reproducibility_checklist.tex
git commit -m "Add schema-v3 robustness study and supporting fixes"
git cherry-pick 163c138 315d71e 2a85ccd 00d8178 32873c9 7eaeaca 7ed8426 3116fc8 609f9e9 7aab3eb cf983d5
```

Cherry-pick new caveat-hardening code commits. Do not cherry-pick `a96d017`. Reconstruct only source `.tex` changes from `5ca6167`; omit results, generated figures, macros, PDFs, PNGs, CSVs, Parquet, checkpoints, and tensors.

- [ ] **Step 3: Prove unique history is code-only**

```powershell
$forbidden = git log --format= --name-only origin/main..HEAD | Sort-Object -Unique | Where-Object {
  $_ -like 'results/reviewer_revision_2026_08/*' -or
  $_ -like 'results/reviewer_caveat_extension_2026_08/*' -or
  $_ -like 'figures/*.png' -or
  $_ -like 'figures/*.pdf' -or
  $_ -eq 'manuscript_numbers.tex' -or
  $_ -match '\.(csv|parquet|pt|npy|png)$' -or
  $_ -eq 'main_revised.pdf'
}
if ($forbidden) { throw "Generated results present in publication history: $($forbidden -join ', ')" }
```

Expected: no forbidden paths. Confirm unrelated deletions are absent from the unique diff and the result history remains at the backup ref.

- [ ] **Step 4: Verify the code-only worktree from scratch**

Run the full unit suite, Ruff, both CLI dry-runs, and README command/path checks in the isolated worktree.

Expected: tests/lint pass; dry runs enumerate required units without local results; fresh-clone paths exist.

- [ ] **Step 5: Dry-run the explicit remote-branch push**

```powershell
git push --dry-run --porcelain origin HEAD:refs/heads/agent/reviewer-revision-2026-08
```

Expected: authentication succeeds and no result payload appears.

- [ ] **Step 6: Push and verify the remote ref**

```powershell
git push --set-upstream origin HEAD:refs/heads/agent/reviewer-revision-2026-08
git ls-remote --heads origin agent/reviewer-revision-2026-08
```

Expected: remote SHA equals local HEAD. Report branch, commits, tests, run ID, paper path/page count, immutable local hashes, remaining limitations, and that generated results were intentionally not pushed.
