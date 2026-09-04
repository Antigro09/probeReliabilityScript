from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.reviewer_revision.figures import (
    FigureValidationError,
    create_construct_panel_figure,
    create_floor_sensitivity_figure,
)
from src.reviewer_revision.paper import (
    ManuscriptValidationError,
    patch_manuscript_with_extension,
    validate_extension_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CELLS = (
    ("pythia", "sva", 6),
    ("pythia", "sst2", 6),
    ("gpt2", "sva", 6),
    ("gpt2", "sst2", 6),
    ("bert", "sva", 6),
    ("bert", "sst2", 6),
    ("qwen", "sva", 14),
    ("qwen", "sst2", 14),
    ("gemma", "sva", 14),
    ("gemma", "sst2", 14),
    ("llama", "sva", 14),
    ("llama", "sst2", 14),
)


def _extension_summary() -> dict[str, object]:
    endpoint_p = 10 / 10_001
    holm_p = 11 * endpoint_p
    cells: dict[str, object] = {}
    for index, (model, task, layer) in enumerate(CELLS):
        slug = f"{model}-{task}-l{layer}"
        pilot = slug == "qwen-sst2-l14"
        endpoints = {
            "median_target_post_edit_accuracy": 0.62 + index / 1000,
            "median_target_recovery_ratio": 0.68 + index / 1000,
            "median_control_retention_ratio": 0.89 - index / 1000,
        }
        record: dict[str, object] = {
            "cell": {
                "model_key": model,
                "model_id": f"organization/{model}",
                "task": task,
                "layer": layer,
            },
            "confirmatory": not pilot,
            "status": "ok",
            "n_groups": 20,
            "n_candidate_edits": 60,
            "endpoints": endpoints,
            "thresholds": {
                "accuracy": 0.55,
                "target_recovery_ratio": 0.50,
                "control_retention_ratio": 0.80,
            },
            "point_threshold_decisions": {
                "accuracy": True,
                "target_recovery_ratio": True,
                "control_retention_ratio": True,
            },
            "passes_locked_point_thresholds": True,
        }
        if pilot:
            record["inference_mode"] = "descriptive_only"
        else:
            record.update(
                {
                    "marginal_lower_bounds": {
                        "accuracy": endpoints["median_target_post_edit_accuracy"] - 0.04,
                        "target_recovery_ratio": endpoints[
                            "median_target_recovery_ratio"
                        ]
                        - 0.05,
                        "control_retention_ratio": endpoints[
                            "median_control_retention_ratio"
                        ]
                        - 0.03,
                    },
                    "marginal_lower_bound_finite": {
                        "accuracy": True,
                        "target_recovery_ratio": True,
                        "control_retention_ratio": True,
                    },
                    "endpoint_p_values": {
                        "accuracy": endpoint_p,
                        "target_recovery_ratio": endpoint_p,
                        "control_retention_ratio": endpoint_p,
                    },
                    "within_cell_combination": (
                        "intersection_union_max_endpoint_p"
                    ),
                    "internal_cell_p_value": endpoint_p,
                    "holm_adjusted_cell_p_value": holm_p,
                    "passes_holm_adjusted_inference": True,
                    "passes_locked_confirmatory_rule": True,
                    "lower_bound_scope": "marginal",
                    "lower_bound_multiplicity_adjusted": False,
                    "lower_bound_simultaneous": False,
                }
            )
        cells[slug] = record

    construct = {
        "schema": "reviewer_revision.construct_panel_inference.v1",
        "schema_version": 1,
        "status": "ok",
        "pilot": cells["qwen-sst2-l14"],
        "confirmatory_cell_count": 11,
        "nonestimable_confirmatory_cell_count": 0,
        "passing_confirmatory_cell_count": 11,
        "cells": cells,
        "bootstrap": {
            "draws": 10_000,
            "seed": 20260830,
            "bit_generator": "PCG64",
            "resampling_unit": "final_test_group_id",
            "shared_resamples_across_models_within_task": True,
        },
        "multiplicity": {
            "method": "holm_one_sided",
            "family_size": 11,
            "alpha": 0.05,
        },
        "lower_bounds": {
            "confidence_level": 0.95,
            "quantile": 0.05,
            "quantile_method": "linear",
            "scope": "marginal",
            "multiplicity_adjusted": False,
            "simultaneous": False,
        },
        "endpoint_tail_p_value": (
            "(1 + count(theta_star <= threshold)) / (B + 1)"
        ),
        "within_cell_combination": "intersection_union_max_endpoint_p",
    }
    partial_center = 0.49808935466666726
    full_center = 0.2756689154117645
    cell_bounds = [
        {
            "model_key": f"model-{index // 10}",
            "task": "sva" if index % 2 == 0 else "sst2",
            "layer": index % 5,
            "planned_pairs": 5,
            "observed_pairs": 4 if index < 9 else 5,
            "missing_pairs": 1 if index < 9 else 0,
            "lower": partial_center - 0.2 if index < 9 else full_center,
            "upper": partial_center + 0.2 if index < 9 else full_center,
        }
        for index in range(60)
    ]
    planned_cells = [
        [row["model_key"], row["task"], row["layer"]]
        for row in cell_bounds
    ]
    floor_curve = []
    for floor, pairs, gap in (
        (0.5000000001, 300, 0.3249261470),
        (0.525, 299, 0.3229437000),
        (0.55, 291, 0.3277103321),
        (0.575, 274, 0.3291076092),
        (0.60, 250, 0.3385586205),
    ):
        excluded_count = 300 - pairs
        full_keys = planned_cells[excluded_count:]
        partial_keys = planned_cells[:excluded_count]
        floor_curve.append(
            {
                "floor": floor,
                "label": "post_hoc_sensitivity",
                "requested_pairs": 300,
                "analyzed_pairs": pairs,
                "pairs": pairs,
                "requested_cells": 60,
                "analyzed_cells": 60,
                "full_cells": len(full_keys),
                "partial_cells": len(partial_keys),
                "missing_cells": 0,
                "full_cell_keys": full_keys,
                "partial_cell_keys": partial_keys,
                "missing_cell_keys": [],
                "requested_model_task_blocks": 12,
                "analyzed_model_task_blocks": 12,
                "matched_mean": gap + 0.1,
                "split_mean": 0.1,
                "gap": gap,
                "equal_block": {
                    "matched": gap + 0.1,
                    "split": 0.1,
                    "gap": gap,
                },
                "excluded_pair_keys": [
                    [*planned_cells[index], 0]
                    for index in range(excluded_count)
                ],
                "excluded_pairs": [
                    {
                        "pair_key": [*planned_cells[index], 0],
                        "reasons": ["fixture_target_acc_pre_below_floor"],
                    }
                    for index in range(excluded_count)
                ],
                "exact_sign_flip": {
                    "method": "two_sided_exact_paired_sign_flip",
                    "n_blocks": 12,
                    "nonzero_count": 11,
                    "zero_count": 1,
                    "observed_mean": gap,
                    "permutations": 2048,
                    "extreme_count": 2,
                    "p_value": 0.0009765625,
                    "zero_tolerance": 0.0,
                    "status": "ok",
                },
                "block_gap_sign_counts": {
                    "positive": 11,
                    "zero": 1,
                    "negative": 0,
                },
            }
        )
    robustness = {
        "schema": "reviewer_revision.floor_robustness.v1",
        "schema_version": 1,
        "status": "ok",
        "label": "post_hoc_sensitivity",
        "post_hoc_sensitivity": True,
        "replaces_registered_primary": False,
        "units": {
            "rows": 600,
            "pairs": 300,
            "cells": 60,
            "model_task_blocks": 12,
        },
        "full_case_raw_drop": {
            "label": "post_hoc_sensitivity",
            "post_hoc_sensitivity": True,
            "replaces_registered_primary": False,
            "matched_mean": 0.42557458057439307,
            "split_mean": 0.1771019040855655,
            "gap": 0.24847267648882757,
            "pairs": 300,
            "cells": 60,
            "model_task_blocks": 12,
            "bootstrap": {
                "method": "equal_block_hierarchical_percentile_bootstrap",
                "cluster_unit": "model_key_task",
                "hierarchy": ["model_task", "layer", "pair"],
                "estimand_weighting": "equal_model_task_blocks",
                "draws": 10_000,
                "seed": 20260830,
                "confidence": 0.95,
                "point_estimate": 0.24847267648882757,
                "equal_block_point_estimate": 0.24847267648882757,
                "ci_low": 0.2046648055280495,
                "ci_high": 0.2970943073101083,
                "n_blocks": 12,
                "n_cells": 60,
                "n_pairs": 300,
            },
            "exact_sign_flip": {
                "method": "two_sided_exact_paired_sign_flip",
                "n_blocks": 12,
                "nonzero_count": 12,
                "zero_count": 0,
                "observed_mean": 0.24847267648882757,
                "permutations": 4096,
                "extreme_count": 2,
                "p_value": 0.00048828125,
                "zero_tolerance": 0.0,
                "status": "ok",
            },
            "block_gap_sign_counts": {
                "positive": 12,
                "zero": 0,
                "negative": 0,
            },
        },
        "floor_curve": floor_curve,
        "partial_identification": {
            "label": "deterministic_partial_identification_bound",
            "sensitivity_label": "post_hoc_sensitivity",
            "floor": 0.55,
            "is_confidence_interval": False,
            "available_case_gap": 0.3277103321,
            "missing_pair_gap_domain": [-1.0, 1.0],
            "planned_pairs": 300,
            "observed_pairs": 291,
            "missing_pairs": 9,
            "planned_cells": 60,
            "wholly_missing_cells": 0,
            "fixed_model_task_blocks": 12,
            "lower": 0.2790319813,
            "upper": 0.3390319813,
            "cell_bounds": cell_bounds,
        },
    }
    return {"floor_robustness": robustness, "construct_panel": construct}


def _assert_pdf_png(pdf: Path, png: Path) -> None:
    assert pdf.is_file() and pdf.stat().st_size > 1_000
    assert png.is_file() and png.stat().st_size > 1_000
    with Image.open(png) as image:
        assert image.info.get("dpi", (0, 0))[0] >= 299


def test_extension_figures_render_validated_floor_and_construct_panels(
    tmp_path: Path,
) -> None:
    source = tmp_path / "extension_summary.json"
    source.write_text(json.dumps(_extension_summary()), encoding="utf-8")

    floor = create_floor_sensitivity_figure(source, tmp_path / "figures")
    construct = create_construct_panel_figure(source, tmp_path / "figures")

    assert floor.metadata["post_hoc_sensitivity"] is True
    assert floor.metadata["locked_floor"] == 0.55
    assert construct.metadata["confirmatory_cells"] == 11
    assert construct.metadata["pilot_cell"] == "qwen-sst2-l14"
    assert construct.metadata["bounds_are_marginal_not_simultaneous"] is True
    _assert_pdf_png(floor.pdf_path, floor.png_path)
    _assert_pdf_png(construct.pdf_path, construct.png_path)


def test_extension_reporting_rejects_pilot_leakage_and_missing_family(
    tmp_path: Path,
) -> None:
    summary = _extension_summary()
    construct = summary["construct_panel"]
    construct["pilot"]["confirmatory"] = True  # type: ignore[index]

    with pytest.raises(ManuscriptValidationError, match="pilot"):
        validate_extension_summary(summary)
    source = tmp_path / "bad.json"
    source.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(FigureValidationError, match="pilot"):
        create_construct_panel_figure(source, tmp_path / "figures")

    summary = _extension_summary()
    del summary["construct_panel"]["cells"]["bert-sva-l6"]  # type: ignore[index]
    with pytest.raises(ManuscriptValidationError, match="12 locked cells"):
        validate_extension_summary(summary)


@pytest.mark.parametrize(
    ("mutator", "message"),
    (
        (
            lambda summary: summary["construct_panel"]["pilot"].update(  # type: ignore[index]
                {"internal_cell_p_value": 0.001}
            ),
            "pilot.*inferential",
        ),
        (
            lambda summary: summary["construct_panel"]["cells"][  # type: ignore[index]
                "bert-sva-l6"
            ]["endpoints"].update({"median_target_post_edit_accuracy": 0.1}),
            "point-threshold",
        ),
        (
            lambda summary: summary["construct_panel"]["cells"][  # type: ignore[index]
                "bert-sva-l6"
            ].update(
                {
                    "holm_adjusted_cell_p_value": 1.0,
                    "passes_holm_adjusted_inference": True,
                }
            ),
            "Holm",
        ),
    ),
)
def test_extension_validation_recomputes_inference_instead_of_trusting_flags(
    mutator,
    message: str,
) -> None:
    summary = _extension_summary()
    mutator(summary)

    with pytest.raises(ManuscriptValidationError, match=message):
        validate_extension_summary(summary)


def test_construct_figure_renders_explicit_nonfinite_marginal_bound(
    tmp_path: Path,
) -> None:
    summary = _extension_summary()
    cell = summary["construct_panel"]["cells"]["bert-sva-l6"]  # type: ignore[index]
    cell["marginal_lower_bounds"]["target_recovery_ratio"] = None
    cell["marginal_lower_bound_finite"]["target_recovery_ratio"] = False

    validate_extension_summary(summary)
    figure = create_construct_panel_figure(summary, tmp_path)

    assert figure.metadata["nonfinite_marginal_bounds"] == [
        "bert-sva-l6:target_recovery_ratio"
    ]
    assert figure.metadata["nonestimable_cells_have_no_scientific_x_value"] is True
    _assert_pdf_png(figure.pdf_path, figure.png_path)


@pytest.mark.parametrize(
    ("mutator", "message"),
    (
        (
            lambda summary: summary["floor_robustness"]["floor_curve"][2].update(  # type: ignore[index]
                {"pairs": 999, "analyzed_pairs": 999}
            ),
            "pair counts",
        ),
        (
            lambda summary: summary["floor_robustness"][  # type: ignore[index]
                "partial_identification"
            ].update({"floor": 0.60}),
            "locked floor",
        ),
        (
            lambda summary: summary["floor_robustness"][  # type: ignore[index]
                "partial_identification"
            ].update({"available_case_gap": -0.2}),
            "available-case gap",
        ),
    ),
)
def test_extension_validation_recomputes_floor_schema_invariants(
    mutator,
    message: str,
) -> None:
    summary = _extension_summary()
    mutator(summary)

    with pytest.raises(ManuscriptValidationError, match=message):
        validate_extension_summary(summary)


@pytest.mark.parametrize(
    ("mutator", "message"),
    (
        (
            lambda summary: summary["floor_robustness"][  # type: ignore[index]
                "full_case_raw_drop"
            ].update({"post_hoc_sensitivity": False}),
            "full-case raw drop.*post-hoc",
        ),
        (
            lambda summary: summary["floor_robustness"][  # type: ignore[index]
                "full_case_raw_drop"
            ]["bootstrap"].update({"method": "iid_rows", "draws": 1, "seed": 999}),
            "hierarchical bootstrap",
        ),
        (
            lambda summary: summary["floor_robustness"][  # type: ignore[index]
                "full_case_raw_drop"
            ]["exact_sign_flip"].update({"method": "one_sided", "status": "failed"}),
            "exact sign-flip",
        ),
        (
            lambda summary: summary["floor_robustness"][  # type: ignore[index]
                "partial_identification"
            ].update({"lower": -1.0, "upper": 1.0}),
            "recomputed equal-block bound",
        ),
        (
            lambda summary: summary["floor_robustness"][  # type: ignore[index]
                "partial_identification"
            ]["cell_bounds"][0].update({"lower": 0.31, "upper": 0.31}),
            "cell bound width",
        ),
        (
            lambda summary: summary["floor_robustness"][  # type: ignore[index]
                "partial_identification"
            ]["cell_bounds"][0].update(
                summary["floor_robustness"]["partial_identification"][  # type: ignore[index]
                    "cell_bounds"
                ][1]
            ),
            "unique planned cells",
        ),
    ),
)
def test_extension_validation_recomputes_raw_and_partial_id_claims(
    mutator,
    message: str,
) -> None:
    summary = _extension_summary()
    mutator(summary)

    with pytest.raises(ManuscriptValidationError, match=message):
        validate_extension_summary(summary)


def test_extension_patch_is_result_neutral_and_preserves_required_caveats(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.tex"
    source.write_text(
        """\\documentclass{article}
\\begin{document}
% BEGIN POST-RUN CONSTRUCT-CONTEXT UPDATE
old construct context
% END POST-RUN CONSTRUCT-CONTEXT UPDATE
% BEGIN POST-RUN FLOOR-SENSITIVITY UPDATE
old floor text
% END POST-RUN FLOOR-SENSITIVITY UPDATE
% BEGIN POST-RUN CONSTRUCT-PANEL UPDATE
old construct text
% END POST-RUN CONSTRUCT-PANEL UPDATE
% BEGIN POST-RUN CONSTRUCT-SCOPE UPDATE
old construct scope
% END POST-RUN CONSTRUCT-SCOPE UPDATE
\\end{document}
""",
        encoding="utf-8",
    )

    report = patch_manuscript_with_extension(
        source,
        tmp_path / "patched.tex",
        _extension_summary(),
        macros_path=tmp_path / "extension_numbers.tex",
    )

    patched = report.destination.read_text(encoding="utf-8")
    macros = report.macros_path.read_text(encoding="utf-8")
    assert "post-hoc full-case sensitivity" in patched
    assert "11 untouched middle-layer cells" in patched
    assert "Holm-adjusted" in patched
    assert "does not prove concept erasure" in patched
    assert "not downstream language-model behavior" in patched
    assert "neither multiplicity-adjusted nor simultaneous" in patched
    assert "\\newcommand{\\FloorRawGap}{0.248}" in macros
    assert "\\newcommand{\\ConstructPointPassingCells}{11}" in macros
    assert "\\newcommand{\\ConstructHolmPassingCells}{11}" in macros
    assert "\\newcommand{\\ConstructPassingCells}{11}" in macros


def test_extension_patch_rejects_unlabeled_floor_analysis(tmp_path: Path) -> None:
    summary = _extension_summary()
    summary["floor_robustness"]["label"] = "primary"  # type: ignore[index]
    with pytest.raises(ManuscriptValidationError, match="post-hoc"):
        patch_manuscript_with_extension(
            tmp_path / "missing.tex",
            tmp_path / "out.tex",
            summary,
        )


def test_extension_patch_applies_to_current_workshop_source_without_broad_rewrite(
    tmp_path: Path,
) -> None:
    report = patch_manuscript_with_extension(
        PROJECT_ROOT / "main_revised.tex",
        tmp_path / "main_revised.tex",
        _extension_summary(),
    )

    patched = report.destination.read_text(encoding="utf-8")
    assert patched.count("BEGIN POST-RUN FLOOR-SENSITIVITY UPDATE") == 1
    assert patched.count("BEGIN POST-RUN CONSTRUCT-PANEL UPDATE") == 1
    assert "BEGIN POST-RUN ORIENTATION-REDECODABILITY UPDATE" not in patched
    assert r"\usepackage[dblblindworkshop]{neurips_2026}" in patched
    assert r"\workshoptitle{Linguistic Principles for Foundation Models}" in patched
    assert r"\author{Anonymous Authors}" in patched
    assert r"\section{Scope, responsible use, and conclusion}" in patched
    assert "does not support attributing the archived saturation" in patched
    assert "fails its nonlinear synthetic gate" in patched
    assert "evaluates them in one prespecified cell" not in patched
    assert "remaining cells still lack these endpoints" not in patched
    assert "one-cell result" not in patched
    assert r"Appendix~\ref{app:pilot-compatibility}" in patched
    assert "fresh MLP decoders" in patched


def test_floor_figure_separates_floor_free_raw_drop_from_floor_curve(
    tmp_path: Path,
) -> None:
    figure = create_floor_sensitivity_figure(_extension_summary(), tmp_path)

    assert figure.metadata["full_case_raw_drop_has_separate_no_floor_category"] is True
    assert figure.metadata["floors"] == [0.5000000001, 0.525, 0.55, 0.575, 0.6]
    _assert_pdf_png(figure.pdf_path, figure.png_path)
