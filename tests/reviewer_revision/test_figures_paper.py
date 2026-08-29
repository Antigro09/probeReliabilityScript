from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.reviewer_revision.figures import (
    FigureValidationError,
    create_construct_check_figure,
    create_epsilon_sweep_figure,
    create_expanded_matched_split_figure,
    generate_revision_figures,
)
from src.reviewer_revision.paper import (
    ManuscriptValidationError,
    _epsilon_prose,
    _epsilon_values,
    generate_manuscript_numbers,
    patch_manuscript,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EPSILONS = (
    0.0,
    0.001953125,
    0.00390625,
    0.0078125,
    0.015625,
    0.03125,
    0.0625,
    0.125,
    0.25,
    0.5,
)


def _assert_pdf_and_png(pdf_path: Path, png_path: Path) -> None:
    assert pdf_path.is_file()
    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert png_path.is_file()
    with Image.open(png_path) as image:
        assert image.width >= 1_500
        assert image.height >= 600
        dpi = image.info.get("dpi")
        assert dpi is not None
        assert dpi[0] == pytest.approx(300.0, abs=0.2)
        assert dpi[1] == pytest.approx(300.0, abs=0.2)


def _matched_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    models = ("pythia", "gpt2", "bert", "qwen", "gemma", "llama")
    for model_index, model in enumerate(models):
        for task_index, task in enumerate(("sva", "sst2")):
            for depth_position, layer in enumerate((1, 4, 6, 9, 12), start=1):
                for pair_seed in range(5):
                    split = 0.38 + 0.025 * depth_position + 0.003 * model_index
                    matched = split + 0.24 - 0.01 * task_index
                    for condition, damage in (("matched", matched), ("split", split)):
                        rows.append(
                            {
                                "model": model,
                                "task": task,
                                "layer": layer,
                                "depth_position": depth_position,
                                "pair_seed": pair_seed,
                                "method": "alterrep",
                                "condition": condition,
                                "target_damage_C": damage,
                                "status": "ok",
                            }
                        )
    return pd.DataFrame(rows)


def _epsilon_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_index, model in enumerate(
        ("pythia", "gpt2", "bert", "qwen", "gemma", "llama")
    ):
        for task in ("sva", "sst2"):
            for pair_seed in range(5):
                for method in ("fgsm", "pgd"):
                    for epsilon_index, epsilon in enumerate(EPSILONS):
                        base = 0.0 if epsilon == 0.0 else min(1.0, 0.12 * epsilon_index)
                        split = (
                            0.0
                            if epsilon == 0.0
                            else min(1.0, base + 0.002 * model_index)
                        )
                        matched = min(1.0, split + (0.05 if epsilon else 0.0))
                        for condition, damage in (("matched", matched), ("split", split)):
                            rows.append(
                                {
                                    "model": model,
                                    "task": task,
                                    "layer": 6,
                                    "pair_seed": pair_seed,
                                    "method": method,
                                    "condition": condition,
                                    "epsilon": epsilon,
                                    "target_damage_C": damage,
                                    "status": "ok",
                                }
                            )
    return pd.DataFrame(rows)


def _construct_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for architecture_index, architecture in enumerate(("linear", "mlp", "mka")):
        for seed in range(20):
            edit_id = f"dcand_crossfit:{architecture}:seed-{seed}"
            raw = 0.72 + 0.002 * seed + 0.015 * architecture_index
            oriented = 0.34 + 0.001 * seed + 0.01 * architecture_index
            for family, accuracy in (("fixed", 0.30), ("fresh_linear", 0.78), ("fresh_mlp", 0.81)):
                rows.append(
                    {
                        "edit_id": edit_id,
                        "edit_object": "candidate",
                        "architecture": architecture,
                        "evaluation_family": family,
                        "label": "target",
                        "accuracy": accuracy - 0.001 * seed,
                        "C_raw": raw,
                        "C_orientation_calibrated": oriented,
                        "status": "ok",
                    }
                )
    return pd.DataFrame(rows)


def _runner_construct_rows() -> pd.DataFrame:
    """Mirror the row schema emitted by runner.run_construct_check."""

    rows = _construct_rows().copy()
    rows["edit_id"] = rows.apply(
        lambda row: (
            f"dcand_crossfit-{row['architecture']}-"
            f"seed{int(str(row['edit_id']).rsplit('-', 1)[1])}"
        ),
        axis=1,
    )
    rows["edit_object"] = "dcand_crossfit"
    rows["model_key"] = "qwen"
    rows["model_id"] = "Qwen/Qwen2.5-1.5B"
    rows["task"] = "sst2"
    rows["layer"] = 14
    rows["candidate_seed"] = rows["edit_id"].str.extract(
        r"seed(\d+)$", expand=False
    ).astype(int)
    return rows


def _write_figure_rows(run_dir: Path) -> None:
    _matched_rows().to_csv(run_dir / "matched_split_rows.csv", index=False)
    _epsilon_rows().to_csv(run_dir / "epsilon_sweep_rows.csv", index=False)
    _construct_rows().to_csv(run_dir / "construct_check_rows.csv", index=False)


def test_expanded_figure_is_vector_plus_300_dpi_png_from_saved_rows(tmp_path: Path) -> None:
    rows_path = tmp_path / "matched.csv"
    _matched_rows().to_csv(rows_path, index=False)

    artifacts = create_expanded_matched_split_figure(
        rows_path,
        tmp_path / "figures",
        bootstrap_draws=50,
    )

    assert artifacts.stem == "fig_circularity_expanded"
    _assert_pdf_and_png(artifacts.pdf_path, artifacts.png_path)


def test_epsilon_figure_contains_locked_grid_and_ceiling_companion(tmp_path: Path) -> None:
    rows_path = tmp_path / "epsilon.csv"
    rows = _epsilon_rows()
    rows.to_csv(rows_path, index=False)

    artifacts = create_epsilon_sweep_figure(
        rows_path,
        tmp_path / "figures",
        bootstrap_draws=50,
    )

    assert artifacts.metadata["epsilons"] == list(EPSILONS)
    assert artifacts.metadata["methods"] == ["fgsm", "pgd"]
    assert artifacts.metadata["includes_ceiling_fraction"] is True
    _assert_pdf_and_png(artifacts.pdf_path, artifacts.png_path)

    incomplete_path = tmp_path / "incomplete.csv"
    rows.loc[~((rows["method"] == "pgd") & (rows["epsilon"] == 0.125))].to_csv(
        incomplete_path, index=False
    )
    with pytest.raises(FigureValidationError, match="epsilon grid"):
        create_epsilon_sweep_figure(incomplete_path, tmp_path / "bad")


def test_construct_figure_aligns_sixty_candidate_endpoint_distributions(
    tmp_path: Path,
) -> None:
    rows_path = tmp_path / "construct.csv"
    _construct_rows().to_csv(rows_path, index=False)

    artifacts = create_construct_check_figure(rows_path, tmp_path / "figures")

    assert artifacts.metadata["candidate_edits"] == 60
    assert artifacts.metadata["endpoint_groups"] == [
        "raw_fixed_damage",
        "orientation_calibrated_damage",
        "fresh_decoder_accuracy",
    ]
    _assert_pdf_and_png(artifacts.pdf_path, artifacts.png_path)


def test_construct_figure_accepts_runner_dcand_crossfit_row_schema(
    tmp_path: Path,
) -> None:
    rows_path = tmp_path / "runner-construct.csv"
    rows = _runner_construct_rows()
    rows.to_csv(rows_path, index=False)

    artifacts = create_construct_check_figure(rows_path, tmp_path / "figures")

    assert artifacts.metadata["candidate_edits"] == 60
    assert rows["edit_object"].unique().tolist() == ["dcand_crossfit"]
    assert rows["edit_id"].str.match(
        r"^dcand_crossfit-(linear|mlp|mka)-seed\d+$"
    ).all()


def test_generate_revision_figures_reads_run_artifacts_without_model_imports(
    tmp_path: Path,
) -> None:
    _write_figure_rows(tmp_path)

    generated = generate_revision_figures(
        tmp_path,
        bootstrap_draws=25,
    )

    assert set(generated) == {"matched_split", "epsilon_sweep", "construct_check"}
    source = (PROJECT_ROOT / "src" / "reviewer_revision" / "figures.py").read_text(
        encoding="utf-8"
    )
    assert "import torch" not in source
    assert "transformers" not in source
    assert "set_title(" not in source


def _summary_payload() -> dict[str, object]:
    common = {
        "schema_version": 1,
        "failed_units": 0,
        "raw_row_file_sha256": "a" * 64,
        "generating_git_commit": "b" * 40,
        "warnings": [],
        "caveats": ["Fixed-decoder damage is not evidence of erasure."],
    }
    curves: list[dict[str, object]] = []
    for method in ("fgsm", "pgd"):
        for condition in ("matched", "split"):
            for index, epsilon in enumerate(EPSILONS):
                mean = 0.0 if epsilon == 0.0 else min(
                    1.0, index / 8 + (0.04 if condition == "matched" else 0.0)
                )
                curves.append(
                    {
                        "method": method,
                        "condition": condition,
                        "epsilon": epsilon,
                        "n_cells": 12,
                        "mean_target_damage_C": mean,
                        "fraction_at_C_equal_1": 1.0 if mean == 1.0 else 0.0,
                    }
                )
    return {
        "schema_version": 1,
        "status": "complete",
        "validation": {"complete": True},
        "matched_split": {
            **common,
            "estimand": "paired equal-cell gap with model-task block inference",
            "included_units": {
                "rows": 600,
                "pairs": 300,
                "cells": 60,
                "model_task_blocks": 12,
            },
            "grand_mean_matched": 0.812345,
            "grand_mean_split": 0.543210,
            "grand_mean_gap": 0.269135,
            "bootstrap": {"ci_low": 0.10123, "ci_high": 0.50123},
            "primary_exact_sign_flip": {"p_value": 0.00048828125},
            "confidence_interval_method": "hierarchical_percentile_bootstrap",
            "cluster_unit": "model_task",
            "manuscript_values": {
                "matched": "0.812",
                "split": "0.543",
                "gap": "0.269",
                "ci": "[0.101, 0.501]",
                "p": "0.000488",
            },
        },
        "epsilon_sweep": {
            **common,
            "estimand": "equal-cell curves at every locked epsilon",
            "included_units": {"rows": 2400, "cells": 480, "model_task_blocks": 12},
            "curves": curves,
            "large_nonmonotonic_reversals": [],
            "has_large_nonmonotonic_reversals": False,
            "nonmonotonic_reversal_threshold": 0.10,
            "epsilon_zero_integrity": {"passed": True},
            "confidence_interval_method": "model_task_cluster_bootstrap_for_figures",
            "cluster_unit": "model_task",
        },
        "construct_check": {
            **common,
            "estimand": "calibration-frozen and fresh-decoder endpoints",
            "included_units": {"rows": 390, "edits": 65},
            "orientation_choice_split": "orientation_calibration",
            "orientation_application_split": "final_test",
            "confidence_interval_method": "descriptive_prespecified_cell",
            "cluster_unit": "edit",
            "manuscript_values": {
                "cell": "Qwen2.5-1.5B/SST-2/layer 14",
                "candidate_edits": 60,
                "raw_median_damage": "0.741",
                "orientation_median_damage": "0.352",
                "fresh_linear_median_accuracy": "0.774",
                "fresh_mlp_median_accuracy": "0.803",
                "fresh_linear_unedited_baseline_accuracy": "0.846",
                "fresh_mlp_unedited_baseline_accuracy": "0.861",
                "fresh_decoder_inference": "descriptive_only",
                "interpretation": "inversion",
            },
        },
    }


def test_epsilon_budget_branch_does_not_require_monotone_damage() -> None:
    summary = _summary_payload()
    epsilon = summary["epsilon_sweep"]  # type: ignore[index]
    curves = epsilon["curves"]  # type: ignore[index]
    for row in curves:
        if (
            row["method"] == "fgsm"
            and row["condition"] == "split"
            and row["epsilon"] == 0.0625
        ):
            row["mean_target_damage_C"] = 0.55

    values = _epsilon_values(epsilon, "fgsm")

    assert values["budget_branch"] == "supports"


def test_epsilon_prose_reports_large_reversals_separately() -> None:
    summary = _summary_payload()
    epsilon = summary["epsilon_sweep"]  # type: ignore[index]
    epsilon.update(  # type: ignore[union-attr]
        {
            "has_large_nonmonotonic_reversals": True,
            "nonmonotonic_reversal_threshold": 0.10,
            "large_nonmonotonic_reversals": [
                {
                    "method": "fgsm",
                    "condition": "split",
                    "epsilon_from": 0.03125,
                    "epsilon_to": 0.0625,
                    "reversal_size": 0.12,
                }
            ],
        }
    )

    prose = _epsilon_prose(summary)

    assert "flags 1 large nonmonotonic reversal" in prose
    assert "reported separately from the perturbation-budget interpretation" in prose


def test_manuscript_macros_are_generated_only_from_validated_summary(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "analysis_summary.json"
    summary_path.write_text(json.dumps(_summary_payload()), encoding="utf-8")
    output_path = tmp_path / "manuscript_numbers.tex"

    generated = generate_manuscript_numbers(summary_path, output_path)

    assert generated == output_path
    text = output_path.read_text(encoding="utf-8")
    assert r"\newcommand{\SplitCells}{60}" in text
    assert r"\newcommand{\SplitBlocks}{12}" in text
    assert r"\newcommand{\SplitPairs}{300}" in text
    assert r"\newcommand{\SplitMatched}{0.812}" in text
    assert r"\newcommand{\SplitCI}{$[0.101, 0.501]$}" in text
    assert r"\newcommand{\ConstructCandidates}{60}" in text
    assert r"\newcommand{\ConstructFreshLinear}{0.774}" in text
    assert r"\newcommand{\ConstructFreshLinearBaseline}{0.846}" in text
    assert r"\newcommand{\ConstructFreshMLPBaseline}{0.861}" in text
    assert "BugFiltered" not in text


def test_manuscript_generation_refuses_incomplete_or_unproven_summaries(
    tmp_path: Path,
) -> None:
    summary = _summary_payload()
    summary["validation"] = {"complete": False}
    output_path = tmp_path / "should_not_exist.tex"

    with pytest.raises(ManuscriptValidationError, match="complete"):
        generate_manuscript_numbers(summary, output_path)
    assert not output_path.exists()

    summary = _summary_payload()
    del summary["construct_check"]["manuscript_values"]  # type: ignore[index]
    with pytest.raises(ManuscriptValidationError, match="manuscript_values"):
        generate_manuscript_numbers(summary, output_path)
    assert not output_path.exists()


def test_paper_patch_preserves_bug_macros_and_changes_only_allowlisted_regions(
    tmp_path: Path,
) -> None:
    original = (PROJECT_ROOT / "main_revised.tex").read_text(encoding="utf-8")
    source = tmp_path / "source.tex"
    destination = tmp_path / "patched.tex"
    macros = tmp_path / "manuscript_numbers.tex"
    source.write_text(original, encoding="utf-8")

    report = patch_manuscript(
        source,
        destination,
        _summary_payload(),
        macros_path=macros,
    )

    patched = destination.read_text(encoding="utf-8")
    assert report.changed_regions == (
        "numeric_macros",
        "matched_split_scope",
        "epsilon_marker",
        "expanded_figure",
        "orientation_marker",
        "construct_scope",
        "appendix_scope",
    )
    for macro in (
        r"\newcommand{\BugFilteredN}{2{,}000}",
        r"\newcommand{\BugFilteredMedian}{0.998}",
        r"\newcommand{\BugFilteredCeiling}{26\%}",
        r"\newcommand{\BugFilteredAlterRep}{72\%}",
    ):
        assert macro in original
        assert macro in patched
    assert "figures/fig_circularity_expanded.pdf" in patched
    assert "figures/fig_epsilon_sweep.pdf" in patched
    assert "figures/fig_orientation_redecodability.pdf" in patched
    assert "60 model--layer--task cells nested in 12 model--task blocks" in patched
    assert "one prespecified model--layer--task cell" in patched
    assert "do not establish erasure elsewhere" in patched
    assert "No materiality threshold or inferential recovery rule was predeclared" in patched
    assert patched.count("BEGIN POST-RUN EPSILON-SWEEP UPDATE") == 1
    assert patched.count("BEGIN POST-RUN ORIENTATION-REDECODABILITY UPDATE") == 1
    assert "\n\\title{Evaluator Reuse Inflates Intervention Scores" in patched
    assert macros.is_file()


def test_paper_patch_refuses_a_draft_missing_required_markers(tmp_path: Path) -> None:
    source = tmp_path / "source.tex"
    source.write_text("\\documentclass{article}\n", encoding="utf-8")

    with pytest.raises(ManuscriptValidationError, match="marker"):
        patch_manuscript(source, tmp_path / "patched.tex", _summary_payload())
    assert not (tmp_path / "patched.tex").exists()
