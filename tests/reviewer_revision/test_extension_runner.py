from __future__ import annotations

import json
import os
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.reviewer_revision import extension_runner
from src.reviewer_revision.artifacts import RunContext, atomic_write_json
from src.reviewer_revision.config import load_revision_config
from src.reviewer_revision.extension_config import load_extension_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTENSION_SPEC = PROJECT_ROOT / "revision_caveat_extension_spec.yaml"
BASE_SPEC = PROJECT_ROOT / "revision_experiment_spec.yaml"


@pytest.fixture
def extension_config():
    return load_extension_config(EXTENSION_SPEC)


@pytest.fixture
def base_config():
    return load_revision_config(BASE_SPEC)


def _context(tmp_path: Path) -> RunContext:
    return RunContext.create(
        output_root=tmp_path,
        config_hash="a" * 64,
        git_commit="b" * 40,
        timestamp=datetime(2026, 8, 31, tzinfo=timezone.utc),
    )


def test_stage_contract_and_dry_run_counts(extension_config, base_config) -> None:
    assert extension_runner.STAGES == (
        "preflight",
        "robustness",
        "construct-panel",
        "analyze",
        "package-artifacts",
        "figures",
        "patch-paper",
    )

    plan = extension_runner.build_dry_run_plan(extension_config, base_config)

    assert plan["confirmatory_cells"] == 11
    assert plan["pilot_cells"] == 1
    assert plan["worker_edits_per_cell"] == 65
    assert plan["total_worker_edits"] == 715
    assert plan["inferential_candidate_edits"] == 660
    assert plan["compatibility_rows"] == 4290
    assert plan["bootstrap_draws"] == 10_000
    assert plan["computes_confirmatory_endpoints"] is False


def test_construct_disk_gate_requires_projection_plus_reserve() -> None:
    gib = 2**30
    extension_runner.require_construct_disk_capacity(
        free_bytes=20 * gib,
        projected_bytes=12 * gib,
        reserve_bytes=8 * gib,
    )
    with pytest.raises(RuntimeError, match="construct disk gate failed"):
        extension_runner.require_construct_disk_capacity(
            free_bytes=20 * gib - 1,
            projected_bytes=12 * gib,
            reserve_bytes=8 * gib,
        )
    with pytest.raises(TypeError, match="integer byte counts"):
        extension_runner.require_construct_disk_capacity(
            free_bytes=True,
            projected_bytes=12 * gib,
            reserve_bytes=8 * gib,
        )


def test_reproduced_base_commit_requires_an_explicit_opt_in() -> None:
    locked = "7aab3eb9f3145da17c96ea05020353eef48904a4"
    reproduced = "a" * 40

    assert extension_runner.base_provenance_mode(locked) == "registered"
    with pytest.raises(RuntimeError, match="generating commit differs"):
        extension_runner.base_provenance_mode(reproduced)
    assert (
        extension_runner.base_provenance_mode(
            reproduced, allow_reproduced_base=True
        )
        == "reproduced"
    )
    with pytest.raises(RuntimeError, match="lowercase hexadecimal"):
        extension_runner.base_provenance_mode(
            "not-a-commit", allow_reproduced_base=True
        )


def test_auto_device_prefers_cuda_then_mps_then_cpu(monkeypatch) -> None:
    monkeypatch.setattr(extension_runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        extension_runner.torch.backends.mps, "is_available", lambda: True
    )
    assert str(extension_runner.select_extension_device("auto")) == "cuda"

    monkeypatch.setattr(extension_runner.torch.cuda, "is_available", lambda: False)
    assert str(extension_runner.select_extension_device("auto")) == "mps"

    monkeypatch.setattr(
        extension_runner.torch.backends.mps, "is_available", lambda: False
    )
    assert str(extension_runner.select_extension_device("auto")) == "cpu"
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        extension_runner.select_extension_device("cuda")


def test_cuda_reproducibility_sets_deterministic_workspace_before_base_setup(
    monkeypatch, base_config
) -> None:
    observed: list[str | None] = []
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(
        extension_runner,
        "configure_base_reproducibility",
        lambda config: observed.append(os.environ.get("CUBLAS_WORKSPACE_CONFIG")),
    )

    extension_runner.configure_extension_reproducibility(
        base_config,
        extension_runner.torch.device("cuda"),
    )

    assert observed == [":4096:8"]


def test_resume_resolution_binds_exact_spec_hash_and_commit(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    wrong_commit = root / "20260830T000000Z-ccccccc"
    exact = root / "20260831T000000Z-bbbbbbb"
    wrong_hash = root / "20260901T000000Z-bbbbbbb"
    for directory, config_hash, commit in (
        (wrong_commit, "a" * 64, "c" * 40),
        (exact, "a" * 64, "b" * 40),
        (wrong_hash, "d" * 64, "b" * 40),
    ):
        directory.mkdir(parents=True)
        atomic_write_json(
            directory / "run_manifest.json",
            {
                "schema_version": 1,
                "run_id": directory.name,
                "config_hash": config_hash,
                "git_commit": commit,
            },
        )

    assert extension_runner.resolve_extension_resume_directory(
        root,
        config_hash="a" * 64,
        git_commit="b" * 40,
    ) == exact
    with pytest.raises(FileNotFoundError, match="exact config hash and Git commit"):
        extension_runner.resolve_extension_resume_directory(
            root,
            config_hash="a" * 64,
            git_commit="e" * 40,
        )


def test_planned_alterrep_keys_are_the_exact_registered_universe(base_config) -> None:
    keys = extension_runner.planned_alterrep_keys(base_config)

    assert len(keys) == 600
    assert len(set(keys)) == 600
    assert {key[4] for key in keys} == {"alterrep"}
    assert {key[5] for key in keys} == {"matched", "split"}


def test_cache_preflight_regenerates_only_known_padding_bug(
    monkeypatch, extension_config, base_config
) -> None:
    cell = next(
        item for item in extension_config.confirmatory_cells if item.model_key == "gemma"
    )
    reconstruction = SimpleNamespace()
    calls: list[tuple[str, int]] = []
    regenerated: list[dict[str, object]] = []
    state = {"fixed": False}

    class Cache:
        def __init__(self, tag: str) -> None:
            self.n_examples = 3
            self.hidden_size = 4
            self.mean_feature_variance = 1.0
            self.class_conditioned_variance = {"0": 1.0, "1": 1.0}
            self.legacy_dtype_semantics = False
            self.selection = SimpleNamespace(
                cache_sha256=(tag[0] * 64),
                data_hash=("d" * 64),
                path=PROJECT_ROOT / "cache" / f"{tag}.pt",
                provenance={
                    "extraction_code_version": (
                        "last-nonpadding-mask-index-v2" if state["fixed"] else "v1"
                    )
                },
            )

    monkeypatch.setattr(
        extension_runner, "reconstruct_tasks", lambda config: {cell.task: reconstruction}
    )

    def load_cache(**kwargs):
        calls.append((str(kwargs["tag"]), int(kwargs["layer"])))
        return Cache(str(kwargs["tag"]))

    monkeypatch.setattr(extension_runner, "load_construct_cache", load_cache)
    monkeypatch.setattr(
        extension_runner,
        "cache_requires_padding_fix",
        lambda model_key, cache: model_key == "gemma" and not state["fixed"],
    )

    def regenerate(**kwargs):
        regenerated.append(kwargs)
        state["fixed"] = True
        return {"status": "regenerated_and_pending_revalidation"}

    monkeypatch.setattr(extension_runner, "regenerate_construct_cache_group", regenerate)

    report = extension_runner.validate_construct_cache_pins(
        extension_config,
        base_config,
        device=extension_runner.torch.device("cpu"),
        cells=(cell,),
    )

    assert len(report["caches"]) == 3
    assert len(regenerated) == 1
    assert set(regenerated[0]["requirements"]) == {
        ("cand", cell.layer),
        ("eval", cell.layer),
        ("inter", cell.layer),
    }
    # The first legacy cache triggers group repair; all three are then reloaded.
    assert len(calls) == 4


def test_cache_preflight_does_not_regenerate_unknown_validation_failure(
    monkeypatch, extension_config, base_config
) -> None:
    cell = extension_config.confirmatory_cells[0]
    monkeypatch.setattr(
        extension_runner, "reconstruct_tasks", lambda config: {cell.task: object()}
    )
    monkeypatch.setattr(
        extension_runner,
        "load_construct_cache",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("corrupt tensor")),
    )
    monkeypatch.setattr(
        extension_runner,
        "regenerate_construct_cache_group",
        lambda **kwargs: pytest.fail("unknown failures must not trigger regeneration"),
    )

    with pytest.raises(ValueError, match="corrupt tensor"):
        extension_runner.validate_construct_cache_pins(
            extension_config,
            base_config,
            device=extension_runner.torch.device("cpu"),
            cells=(cell,),
        )


def test_cache_pin_records_tensor_and_provenance_sidecar(
    monkeypatch, tmp_path: Path, extension_config
) -> None:
    monkeypatch.setattr(extension_runner, "PROJECT_ROOT", tmp_path)
    tensor = tmp_path / "cache.pt"
    metadata = tmp_path / "cache.json"
    tensor.write_bytes(b"tensor")
    metadata.write_text('{"schema_version": 1}\n', encoding="utf-8")
    cache = SimpleNamespace(
        n_examples=2,
        hidden_size=3,
        mean_feature_variance=1.0,
        class_conditioned_variance={"0": 1.0, "1": 1.0},
        legacy_dtype_semantics=False,
        selection=SimpleNamespace(
            path=tensor,
            data_hash="d" * 64,
            cache_sha256=extension_runner.sha256_file(tensor),
            provenance={"extraction_code_version": "last-nonpadding-mask-index-v2"},
        ),
    )

    record = extension_runner._cache_record(
        cache,
        extension_config.confirmatory_cells[0],
        "cand",
    )

    assert record["metadata_path"] == metadata.name
    assert record["metadata_sha256"] == extension_runner.sha256_file(metadata)


def test_all_stops_immediately_when_a_stage_result_is_invalid(
    monkeypatch, tmp_path: Path, extension_config, base_config
) -> None:
    called: list[str] = []

    monkeypatch.setattr(extension_runner, "current_git_commit", lambda: "b" * 40)
    monkeypatch.setattr(
        extension_runner,
        "select_extension_device",
        lambda requested: extension_runner.torch.device("cpu"),
    )

    def run_stage(stage, *args, **kwargs):
        called.append(stage)
        if stage == "preflight":
            return {"status": "ok"}
        return {"status": "failed"}

    monkeypatch.setattr(extension_runner, "run_stage", run_stage)
    monkeypatch.setattr(extension_runner, "stage_is_complete", lambda *a, **k: False)
    args = Namespace(
        output_root=tmp_path / "runs",
        resume=False,
        device="cpu",
        config=EXTENSION_SPEC,
        base_config=BASE_SPEC,
    )

    with pytest.raises(RuntimeError, match="robustness returned status 'failed'"):
        extension_runner.execute("all", extension_config, base_config, args)

    assert called == ["preflight", "robustness"]
    manifests = list((tmp_path / "runs").glob("*/run_manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["pipeline"]["status"] == "failed"
    assert manifest["pipeline"]["last_completed_stage"] == "preflight"


def test_analyze_requires_validated_twelve_cell_rows_before_inference(
    monkeypatch, tmp_path: Path, extension_config, base_config
) -> None:
    context = _context(tmp_path)
    rows_path = context.run_dir / "construct_panel_group_rows.parquet"
    rows_path.write_bytes(b"rows")
    robustness_path = context.run_dir / "floor_robustness_summary.json"
    atomic_write_json(robustness_path, {"status": "ok", "value": 1})
    atomic_write_json(
        context.run_dir / "construct_panel_report.json",
        {
            "status": "ok",
            "group_rows": {
                "path": rows_path.name,
                "sha256": extension_runner.sha256_file(rows_path),
            },
        },
    )
    order: list[str] = []

    monkeypatch.setattr(
        extension_runner,
        "validate_construct_group_rows_without_inference",
        lambda rows, config: order.append("validate") or {"blocked": {}},
    )
    monkeypatch.setattr(
        extension_runner,
        "analyze_construct_panel",
        lambda rows, *, config: order.append("analyze")
        or {"status": "ok", "confirmatory_cell_count": 11},
    )

    result = extension_runner.run_analyze(context, extension_config, base_config)

    assert order == ["validate", "analyze"]
    assert result["status"] == "complete"
    assert result["construct_panel"]["confirmatory_cell_count"] == 11
    context.close()


def test_packaged_manifest_stays_valid_after_stage_manifest_update(
    tmp_path: Path, extension_config, base_config
) -> None:
    context = _context(tmp_path)
    required = (
        "preflight_report.json",
        "floor_robustness_summary.json",
        "construct_panel_report.json",
        "construct_panel_group_rows.parquet",
        "construct_panel_group_rows.csv",
    )
    for name in required:
        (context.run_dir / name).write_text("payload\n", encoding="utf-8")
    atomic_write_json(
        context.run_dir / "extension_analysis_summary.json",
        {"status": "complete"},
    )

    result = extension_runner.run_package_artifacts(
        context,
        extension_config,
        base_config,
        device=extension_runner.torch.device("cpu"),
    )

    assert result["status"] == "ok"
    assert extension_runner.stage_is_complete(
        context,
        "package-artifacts",
        extension=extension_config,
        base=base_config,
    )
    context.close()


def test_completed_construct_cell_requires_every_materialized_artifact_hash(
    tmp_path: Path, extension_config
) -> None:
    context = _context(tmp_path)
    cell = extension_config.confirmatory_cells[0]
    compatibility = context.run_dir / "compatibility.parquet"
    groups = context.run_dir / "groups.parquet"
    compatibility.write_bytes(b"compatibility")
    groups.write_bytes(b"groups")
    context.write_json_shard(
        "construct-panel-cells",
        (cell.model_key, cell.task, cell.layer),
        {
            "status": "ok",
            "cell_slug": cell.slug,
            "compatibility_rows": {
                "path": compatibility.name,
                "sha256": extension_runner.sha256_file(compatibility),
            },
            "group_rows": {
                "path": groups.name,
                "sha256": extension_runner.sha256_file(groups),
            },
        },
    )

    with pytest.raises(TypeError, match="lacks compatibility_csv"):
        extension_runner._validate_completed_cell_shard(context, cell)
    context.close()


def test_workshop_pdf_accepts_exact_fifty_megabyte_limit(
    monkeypatch, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(extension_runner, "PROJECT_ROOT", project)
    (project / "main_revised.tex").write_text(
        """\\usepackage[dblblindworkshop]{neurips_2026}
\\workshoptitle{Linguistic Principles for Foundation Models}
\\author{Anonymous Authors}
\\section{Scope, responsible use, and conclusion}
\\bibliography{references}
\\appendix
\\input{neurips_2026_checklist.tex}
""",
        encoding="utf-8",
    )
    checklist = [r"\section*{NeurIPS Paper Checklist}"]
    checklist.extend([r"\item[] Answer: \answerYes{}"] * 16)
    (project / "neurips_2026_checklist.tex").write_text(
        "\n".join(checklist), encoding="utf-8"
    )
    context = _context(tmp_path / "runs")
    pdf = context.run_dir / "paper" / "main_revised.pdf"
    pdf.parent.mkdir()
    with pdf.open("wb") as handle:
        handle.seek(50 * 1024 * 1024 - 1)
        handle.write(b"x")
    monkeypatch.setattr(
        extension_runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=b"References\nNeurIPS Paper Checklist\n",
        ),
    )

    result = extension_runner.validate_workshop_submission(
        context,
        {
            "status": "compiled",
            "main_text_pages": 9,
            "visual_inspection": "pending",
            "pdf_sha256": extension_runner.sha256_file(pdf),
        },
    )

    assert result["pdf_bytes"] == 50 * 1024 * 1024
    context.close()


def test_direct_later_stage_refuses_missing_predecessors(
    monkeypatch, tmp_path: Path, extension_config, base_config
) -> None:
    monkeypatch.setattr(extension_runner, "current_git_commit", lambda: "b" * 40)
    monkeypatch.setattr(
        extension_runner,
        "select_extension_device",
        lambda requested: extension_runner.torch.device("cpu"),
    )
    args = Namespace(
        output_root=tmp_path / "runs",
        resume=False,
        device="cpu",
        config=EXTENSION_SPEC,
        base_config=BASE_SPEC,
    )

    with pytest.raises(RuntimeError, match="requires completed predecessor preflight"):
        extension_runner.execute(
            "construct-panel", extension_config, base_config, args
        )


def test_patch_paper_pipeline_finishes_pending_visual_inspection() -> None:
    assert extension_runner._pipeline_final_status("figures") == "stage_complete"
    assert (
        extension_runner._pipeline_final_status("patch-paper")
        == "pending_visual_inspection"
    )
