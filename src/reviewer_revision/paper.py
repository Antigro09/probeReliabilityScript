"""Validated manuscript-number generation and allowlisted LaTeX patching."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

LOCKED_EPSILONS = (
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
SECTION_ALIASES = {
    "matched_split": ("matched_split", "matched_split_summary"),
    "epsilon_sweep": ("epsilon_sweep", "epsilon_sweep_summary"),
    "construct_check": ("construct_check", "construct_check_summary"),
}
LOCKED_EXTENSION_CELLS = frozenset(
    {
        "pythia-sva-l6",
        "pythia-sst2-l6",
        "gpt2-sva-l6",
        "gpt2-sst2-l6",
        "bert-sva-l6",
        "bert-sst2-l6",
        "qwen-sva-l14",
        "qwen-sst2-l14",
        "gemma-sva-l14",
        "gemma-sst2-l14",
        "llama-sva-l14",
        "llama-sst2-l14",
    }
)
LOCKED_EXTENSION_PILOT = "qwen-sst2-l14"
LOCKED_FLOOR_GRID = (0.5000000001, 0.525, 0.55, 0.575, 0.60)
LOCKED_CONSTRUCT_THRESHOLDS = {
    "accuracy": 0.55,
    "target_recovery_ratio": 0.50,
    "control_retention_ratio": 0.80,
}
CONSTRUCT_ENDPOINT_FIELDS = {
    "accuracy": "median_target_post_edit_accuracy",
    "target_recovery_ratio": "median_target_recovery_ratio",
    "control_retention_ratio": "median_control_retention_ratio",
}
PILOT_FORBIDDEN_INFERENCE_FIELDS = frozenset(
    {
        "endpoint_p_values",
        "internal_cell_p_value",
        "marginal_lower_bounds",
        "marginal_lower_bound_finite",
        "within_cell_combination",
        "task_resample_sha256",
        "lower_bound_scope",
        "lower_bound_multiplicity_adjusted",
        "lower_bound_simultaneous",
        "holm_adjusted_cell_p_value",
        "passes_holm_adjusted_inference",
        "passes_locked_confirmatory_rule",
    }
)


class ManuscriptValidationError(ValueError):
    """Raised before any paper output when analysis or markers are incomplete."""


@dataclass(frozen=True)
class PaperPatchReport:
    """Audit record for an allowlisted manuscript patch."""

    destination: Path
    macros_path: Path
    changed_regions: tuple[str, ...]


def _load_summary(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManuscriptValidationError(f"could not load analysis summary: {exc}") from exc
    if not isinstance(payload, dict):
        raise ManuscriptValidationError("analysis_summary.json must contain an object")
    return payload


def _section(payload: Mapping[str, Any], canonical: str) -> dict[str, Any]:
    for key in SECTION_ALIASES[canonical]:
        value = payload.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    raise ManuscriptValidationError(
        f"analysis summary lacks the validated {canonical!r} section"
    )


def _require_mapping(parent: Mapping[str, Any], key: str, location: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ManuscriptValidationError(f"{location}.{key} must be a mapping")
    return dict(value)


def _require_number(parent: Mapping[str, Any], key: str, location: str) -> float:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManuscriptValidationError(f"{location}.{key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ManuscriptValidationError(f"{location}.{key} must be finite")
    return number


def _require_nonempty(parent: Mapping[str, Any], key: str, location: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManuscriptValidationError(f"{location}.{key} must be non-empty")
    return value


def _validate_common(section: Mapping[str, Any], name: str) -> None:
    if section.get("schema_version") != 1:
        raise ManuscriptValidationError(f"{name}.schema_version must be 1")
    if section.get("failed_units") != 0:
        raise ManuscriptValidationError(f"{name} is incomplete because failed_units is nonzero")
    _require_mapping(section, "included_units", name)
    _require_nonempty(section, "estimand", name)
    _require_nonempty(section, "confidence_interval_method", name)
    _require_nonempty(section, "cluster_unit", name)
    digest = _require_nonempty(section, "raw_row_file_sha256", name)
    if re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
        raise ManuscriptValidationError(f"{name}.raw_row_file_sha256 is not SHA-256")
    commit = _require_nonempty(section, "generating_git_commit", name)
    if re.fullmatch(r"[0-9a-fA-F]{7,64}", commit) is None:
        raise ManuscriptValidationError(f"{name}.generating_git_commit is invalid")
    for key in ("warnings", "caveats"):
        value = section.get(key)
        if not isinstance(value, list):
            raise ManuscriptValidationError(f"{name}.{key} must be a list")


def _validate_matched(section: Mapping[str, Any]) -> None:
    included = _require_mapping(section, "included_units", "matched_split")
    analyzed = _require_mapping(section, "analyzed_units", "matched_split")
    cells = included.get("cells")
    blocks = included.get("model_task_blocks")
    pairs = included.get("pairs")
    if cells not in (36, 60):
        raise ManuscriptValidationError("matched_split must contain the locked 36 or 60 cells")
    if blocks != 12:
        raise ManuscriptValidationError("matched_split must contain 12 model-task blocks")
    if pairs != int(cells) * 5:
        raise ManuscriptValidationError("matched_split must contain five pairs per cell")
    analyzed_pairs = analyzed.get("pairs")
    score_null_pairs = section.get("primary_score_null_pairs")
    analyzed_cells = analyzed.get("cells")
    score_null_cells = section.get("primary_score_null_cells")
    if (
        not isinstance(analyzed_pairs, int)
        or isinstance(score_null_pairs, bool)
        or not isinstance(score_null_pairs, int)
        or analyzed_pairs <= 0
        or score_null_pairs < 0
        or analyzed_pairs + score_null_pairs != pairs
    ):
        raise ManuscriptValidationError(
            "matched_split requested, analyzed, and score-null pair counts disagree"
        )
    if (
        not isinstance(analyzed_cells, int)
        or isinstance(score_null_cells, bool)
        or not isinstance(score_null_cells, int)
        or analyzed_cells <= 0
        or score_null_cells < 0
        or analyzed_cells + score_null_cells != cells
        or score_null_pairs < 5 * score_null_cells
    ):
        raise ManuscriptValidationError(
            "matched_split requested, analyzed, and score-null cell counts disagree"
        )
    if analyzed.get("rows") != 2 * analyzed_pairs:
        raise ManuscriptValidationError(
            "matched_split analyzed rows must contain both conditions per analyzed pair"
        )
    if analyzed.get("model_task_blocks") != blocks:
        raise ManuscriptValidationError(
            "matched_split score-floor exclusions removed a requested block"
        )
    matched = _require_number(section, "grand_mean_matched", "matched_split")
    split = _require_number(section, "grand_mean_split", "matched_split")
    gap = _require_number(section, "grand_mean_gap", "matched_split")
    if not math.isclose(matched - split, gap, abs_tol=1.0e-12):
        raise ManuscriptValidationError("matched_split grand means do not reproduce the gap")
    bootstrap = _require_mapping(section, "bootstrap", "matched_split")
    low = _require_number(bootstrap, "ci_low", "matched_split.bootstrap")
    high = _require_number(bootstrap, "ci_high", "matched_split.bootstrap")
    if low > high:
        raise ManuscriptValidationError("matched_split bootstrap interval is reversed")
    sign_flip = _require_mapping(
        section, "primary_exact_sign_flip", "matched_split"
    )
    p_value = _require_number(sign_flip, "p_value", "matched_split.primary_exact_sign_flip")
    if not 0.0 <= p_value <= 1.0:
        raise ManuscriptValidationError("matched_split sign-flip p-value is outside [0, 1]")
    values = _require_mapping(section, "manuscript_values", "matched_split")
    expected = {
        "matched": f"{matched:.3f}",
        "split": f"{split:.3f}",
        "gap": f"{gap:.3f}",
        "ci": f"[{low:.3f}, {high:.3f}]",
        "p": f"{p_value:.6f}",
    }
    for key, expected_value in expected.items():
        if values.get(key) != expected_value:
            raise ManuscriptValidationError(
                f"matched_split.manuscript_values.{key} must equal {expected_value!r}"
            )


def _curve_key(row: Mapping[str, Any]) -> tuple[str, str, float]:
    try:
        return (
            str(row["method"]).lower(),
            str(row["condition"]).lower(),
            float(row["epsilon"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ManuscriptValidationError("epsilon curve row has an invalid key") from exc


def _validate_epsilon(section: Mapping[str, Any]) -> None:
    included = _require_mapping(section, "included_units", "epsilon_sweep")
    analyzed = _require_mapping(section, "analyzed_units", "epsilon_sweep")
    score_null_pairs = section.get("score_null_pairs")
    score_null_units = section.get("score_null_units")
    score_null_curve_cells = section.get("score_null_curve_cells")
    if (
        isinstance(score_null_pairs, bool)
        or not isinstance(score_null_pairs, int)
        or isinstance(score_null_units, bool)
        or not isinstance(score_null_units, int)
        or isinstance(score_null_curve_cells, bool)
        or not isinstance(score_null_curve_cells, int)
        or score_null_pairs < 0
        or score_null_units < 0
        or score_null_curve_cells < 0
        or score_null_curve_cells % 2 != 0
        or score_null_pairs < 5 * (score_null_curve_cells // 2)
    ):
        raise ManuscriptValidationError(
            "epsilon_sweep score-null counts must be non-negative integers"
        )
    included_rows = included.get("rows")
    analyzed_rows = analyzed.get("rows")
    analyzed_pairs = analyzed.get("paired_units")
    if (
        not isinstance(included_rows, int)
        or not isinstance(analyzed_rows, int)
        or not isinstance(analyzed_pairs, int)
        or included_rows <= 0
        or analyzed_rows <= 0
        or analyzed_pairs <= 0
    ):
        raise ManuscriptValidationError(
            "epsilon_sweep requested and analyzed unit counts must be positive integers"
        )
    if included_rows - analyzed_rows != 2 * score_null_pairs:
        raise ManuscriptValidationError(
            "epsilon_sweep must exclude both conditions of each score-null pair"
        )
    if analyzed_rows != 2 * analyzed_pairs:
        raise ManuscriptValidationError(
            "epsilon_sweep analyzed rows must contain both conditions per paired unit"
        )
    if score_null_units > 2 * score_null_pairs:
        raise ManuscriptValidationError(
            "epsilon_sweep score-null rows exceed their paired exclusions"
        )
    included_cells = included.get("cells")
    analyzed_cells = analyzed.get("cells")
    if (
        not isinstance(included_cells, int)
        or not isinstance(analyzed_cells, int)
        or included_cells <= 0
        or analyzed_cells <= 0
        or included_cells - analyzed_cells != score_null_curve_cells
    ):
        raise ManuscriptValidationError(
            "epsilon_sweep requested, analyzed, and score-null curve-cell counts disagree"
        )
    zero = _require_mapping(section, "epsilon_zero_integrity", "epsilon_sweep")
    if zero.get("passed") is not True:
        raise ManuscriptValidationError("epsilon_sweep is incomplete because epsilon zero failed")
    curves = section.get("curves")
    if not isinstance(curves, list) or not all(isinstance(row, Mapping) for row in curves):
        raise ManuscriptValidationError("epsilon_sweep.curves must be a list of mappings")
    keys = [_curve_key(row) for row in curves]
    expected = {
        (method, condition, epsilon)
        for method in ("fgsm", "pgd")
        for condition in ("matched", "split")
        for epsilon in LOCKED_EPSILONS
    }
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ManuscriptValidationError("epsilon_sweep curves do not cover the locked grid")
    for row in curves:
        mean = _require_number(row, "mean_target_damage_C", "epsilon_sweep.curves[]")
        fraction = _require_number(row, "fraction_at_C_equal_1", "epsilon_sweep.curves[]")
        if not 0.0 <= mean <= 1.0 or not 0.0 <= fraction <= 1.0:
            raise ManuscriptValidationError("epsilon curve values must lie in [0, 1]")


def _validate_construct(section: Mapping[str, Any]) -> None:
    included = _require_mapping(section, "included_units", "construct_check")
    if included.get("edits") != 65:
        raise ManuscriptValidationError("construct_check must cover 5 AlterRep and 60 candidate edits")
    if included.get("rows") != 390:
        raise ManuscriptValidationError(
            "construct_check must contain 65 edits x 3 evaluator families x 2 labels"
        )
    if section.get("orientation_choice_split") != "orientation_calibration":
        raise ManuscriptValidationError("construct_check orientation must be chosen on calibration")
    if section.get("orientation_application_split") != "final_test":
        raise ManuscriptValidationError("construct_check orientation must be frozen on final test")
    values = _require_mapping(section, "manuscript_values", "construct_check")
    if values.get("candidate_edits") != 60:
        raise ManuscriptValidationError("construct_check.manuscript_values requires 60 candidates")
    _require_nonempty(values, "cell", "construct_check.manuscript_values")
    for key in (
        "raw_median_damage",
        "orientation_median_damage",
        "fresh_linear_median_accuracy",
        "fresh_mlp_median_accuracy",
        "fresh_linear_unedited_baseline_accuracy",
        "fresh_mlp_unedited_baseline_accuracy",
    ):
        text = _require_nonempty(values, key, "construct_check.manuscript_values")
        try:
            number = float(text)
        except ValueError as exc:
            raise ManuscriptValidationError(
                f"construct_check.manuscript_values.{key} must be numeric text"
            ) from exc
        if not 0.0 <= number <= 1.0:
            raise ManuscriptValidationError(
                f"construct_check.manuscript_values.{key} must lie in [0, 1]"
            )
    if values.get("fresh_decoder_inference") != "descriptive_only":
        raise ManuscriptValidationError(
            "construct fresh-decoder endpoints must remain descriptive without a "
            "predeclared materiality rule"
        )
    interpretation = values.get("interpretation")
    if interpretation not in {"inversion", "neither_failure"}:
        raise ManuscriptValidationError(
            "construct_check.manuscript_values.interpretation is not a locked branch"
        )


def validate_analysis_summary(
    source: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Return a normalized composite summary only after every paper gate passes."""

    payload = _load_summary(source)
    if payload.get("schema_version") != 1:
        raise ManuscriptValidationError("analysis summary schema_version must be 1")
    if payload.get("status") != "complete":
        raise ManuscriptValidationError("analysis summary status is not complete")
    validation = _require_mapping(payload, "validation", "analysis_summary")
    if validation.get("complete") is not True:
        raise ManuscriptValidationError("analysis summary validation is not complete")
    normalized: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "validation": validation,
    }
    validators = {
        "matched_split": _validate_matched,
        "epsilon_sweep": _validate_epsilon,
        "construct_check": _validate_construct,
    }
    for name, validator in validators.items():
        section = _section(payload, name)
        _validate_common(section, name)
        validator(section)
        normalized[name] = section
    return normalized


def _curve_lookup(section: Mapping[str, Any]) -> dict[tuple[str, str, float], dict[str, Any]]:
    return {
        _curve_key(row): dict(row)
        for row in section["curves"]
    }


def _epsilon_text(value: float) -> str:
    return "0" if value == 0.0 else f"{value:.9f}".rstrip("0").rstrip(".")


def _epsilon_values(section: Mapping[str, Any], method: str) -> dict[str, str]:
    curves = _curve_lookup(section)
    archived_matched = float(curves[(method, "matched", 0.5)]["mean_target_damage_C"])
    archived_split = float(curves[(method, "split", 0.5)]["mean_target_damage_C"])
    lower_split = [
        (
            epsilon,
            float(curves[(method, "split", epsilon)]["mean_target_damage_C"]),
            float(curves[(method, "split", epsilon)]["fraction_at_C_equal_1"]),
        )
        for epsilon in LOCKED_EPSILONS[1:-1]
    ]
    departing = [
        epsilon
        for epsilon, mean, fraction in lower_split
        if mean < 1.0 - 1.0e-9 or fraction < 1.0 - 1.0e-9
    ]
    exit_epsilon = max(departing) if departing else 0.0
    supports_budget = (
        math.isclose(archived_matched, 1.0, abs_tol=1.0e-9)
        and math.isclose(archived_split, 1.0, abs_tol=1.0e-9)
        and bool(departing)
    )
    return {
        "archived_matched": f"{archived_matched:.3f}",
        "archived_split": f"{archived_split:.3f}",
        "ceiling_exit": _epsilon_text(exit_epsilon),
        "budget_branch": "supports" if supports_budget else "does_not_support",
    }


def _epsilon_reversal_prose(section: Mapping[str, Any]) -> str:
    reversals = section.get("large_nonmonotonic_reversals")
    has_reversals = section.get("has_large_nonmonotonic_reversals")
    threshold = section.get("nonmonotonic_reversal_threshold")
    if not isinstance(reversals, list) or not all(
        isinstance(record, Mapping) for record in reversals
    ):
        raise ManuscriptValidationError(
            "epsilon_sweep.large_nonmonotonic_reversals must be a list of mappings"
        )
    if has_reversals is not bool(reversals):
        raise ManuscriptValidationError(
            "epsilon_sweep reversal flag disagrees with its reversal records"
        )
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ManuscriptValidationError(
            "epsilon_sweep.nonmonotonic_reversal_threshold must be numeric"
        )
    threshold_value = float(threshold)
    if not math.isfinite(threshold_value) or threshold_value < 0.0:
        raise ManuscriptValidationError(
            "epsilon_sweep.nonmonotonic_reversal_threshold must be finite and non-negative"
        )
    count = len(reversals)
    if count:
        noun = "reversal" if count == 1 else "reversals"
        return (
            f"The analysis flags {count} large nonmonotonic {noun} above the "
            f"predeclared {threshold_value:.3f} threshold; this diagnostic is "
            "reported separately from the perturbation-budget interpretation."
        )
    return (
        "No large nonmonotonic reversal exceeds the predeclared "
        f"{threshold_value:.3f} threshold; this diagnostic is reported separately "
        "from the perturbation-budget interpretation."
    )


def _escape_tex_text(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _macro_lines(summary: Mapping[str, Any]) -> list[str]:
    matched = summary["matched_split"]
    included = matched["included_units"]
    manuscript = matched["manuscript_values"]
    epsilon = summary["epsilon_sweep"]
    fgsm = _epsilon_values(epsilon, "fgsm")
    pgd = _epsilon_values(epsilon, "pgd")
    construct = summary["construct_check"]["manuscript_values"]
    definitions: list[tuple[str, str]] = [
        ("SplitCells", str(included["cells"])),
        ("SplitBlocks", str(included["model_task_blocks"])),
        ("SplitPairs", str(included["pairs"])),
        ("SplitAnalyzedCells", str(matched["analyzed_units"]["cells"])),
        ("SplitAnalyzedPairs", str(matched["analyzed_units"]["pairs"])),
        ("SplitScoreNullPairs", str(matched["primary_score_null_pairs"])),
        ("SplitMatched", str(manuscript["matched"])),
        ("SplitIndependent", str(manuscript["split"])),
        ("SplitGap", str(manuscript["gap"])),
        ("SplitCI", f"${manuscript['ci']}$"),
        ("SplitP", str(manuscript["p"])),
        ("FGSMArchivedMatched", fgsm["archived_matched"]),
        ("FGSMArchivedSplit", fgsm["archived_split"]),
        ("FGSMCeilingExit", fgsm["ceiling_exit"]),
        ("PGDArchivedMatched", pgd["archived_matched"]),
        ("PGDArchivedSplit", pgd["archived_split"]),
        ("PGDCeilingExit", pgd["ceiling_exit"]),
        ("ConstructCell", _escape_tex_text(str(construct["cell"]))),
        ("ConstructCandidates", str(construct["candidate_edits"])),
        ("ConstructRawMedian", str(construct["raw_median_damage"])),
        ("ConstructOrientedMedian", str(construct["orientation_median_damage"])),
        ("ConstructFreshLinear", str(construct["fresh_linear_median_accuracy"])),
        ("ConstructFreshMLP", str(construct["fresh_mlp_median_accuracy"])),
        (
            "ConstructFreshLinearBaseline",
            str(construct["fresh_linear_unedited_baseline_accuracy"]),
        ),
        (
            "ConstructFreshMLPBaseline",
            str(construct["fresh_mlp_unedited_baseline_accuracy"]),
        ),
    ]
    return [rf"\newcommand{{\{name}}}{{{value}}}" for name, value in definitions]


def _atomic_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def generate_manuscript_numbers(
    analysis_summary: str | Path | Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Generate LaTeX macros only from a complete, internally checked summary."""

    summary = validate_analysis_summary(analysis_summary)
    text = "% Generated from validated analysis_summary.json; do not edit.\n"
    text += "\n".join(_macro_lines(summary)) + "\n"
    return _atomic_text(Path(output_path), text)


def _replace_exact(text: str, old: str, new: str, label: str, count: int = 1) -> str:
    observed = text.count(old)
    if observed != count:
        raise ManuscriptValidationError(
            f"expected {count} exact {label} target(s), observed {observed}"
        )
    return text.replace(old, new)


def _replace_marker(text: str, name: str, replacement: str) -> str:
    begin = f"% BEGIN POST-RUN {name} UPDATE"
    end = f"% END POST-RUN {name} UPDATE"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ManuscriptValidationError(f"required {name} marker pair is missing or duplicated")
    start = text.index(begin)
    finish = text.index(end, start) + len(end)
    if finish <= start:
        raise ManuscriptValidationError(f"required {name} marker order is invalid")
    return text[:start] + begin + "\n" + replacement.strip() + "\n" + end + text[finish:]


def _epsilon_prose(summary: Mapping[str, Any]) -> str:
    section = summary["epsilon_sweep"]
    fgsm = _epsilon_values(section, "fgsm")
    pgd = _epsilon_values(section, "pgd")
    both_support = fgsm["budget_branch"] == pgd["budget_branch"] == "supports"
    if both_support:
        interpretation = (
            "For both attacks, split-evaluator damage leaves the ceiling at one or more "
            "prespecified lower budgets while the archived budget remains at the ceiling. "
            "This pattern is consistent with perturbation magnitude contributing to the "
            "archived saturation; it does not identify attack strength as the only cause."
        )
    else:
        interpretation = (
            "The prespecified curves do not jointly show an archived-budget ceiling and "
            "split-evaluator damage leaving that ceiling at a lower budget. The sweep "
            "therefore does not support attributing the archived saturation to "
            "perturbation magnitude alone."
        )
    reversal_diagnostic = _epsilon_reversal_prose(section)
    score_null_pairs = int(section["score_null_pairs"])
    score_null_units = int(section["score_null_units"])
    score_null_curve_cells = int(section["score_null_curve_cells"])
    floor_note = ""
    if score_null_pairs:
        floor_note = (
            f" The locked denominator floor produced {score_null_units} null score rows "
            f"across {score_null_pairs} paired epsilon units; both scoring conditions "
            "of every affected unit remain in the row artifact and are excluded "
            "symmetrically from curve estimates. This makes "
            f"{score_null_curve_cells} requested curve cells non-estimable; all "
            "model--task blocks remain represented."
        )
    return rf"""\paragraph{{Perturbation-budget sweep.}}
At the archived $\epsilon=0.5$ budget, FGSM has matched and split damage \FGSMArchivedMatched{{}} and \FGSMArchivedSplit{{}}, while PGD has \PGDArchivedMatched{{}} and \PGDArchivedSplit{{}}. {interpretation} {reversal_diagnostic}{floor_note}

\begin{{figure}}[t]
  \centering
  \includegraphics[width=\linewidth]{{figures/fig_epsilon_sweep.pdf}}
  \caption{{FGSM and PGD over the complete prespecified $\epsilon$ grid. Curves show equal-cell mean target damage with 95\% model--task cluster intervals; the companion rows show the fraction of pair units at $C=1$. The dotted vertical line marks the archived $\epsilon=0.5$ setting, and zero is the separately plotted no-op point.}}
  \label{{fig:epsilon-sweep}}
\end{{figure}}"""


def _construct_prose(summary: Mapping[str, Any]) -> str:
    values = summary["construct_check"]["manuscript_values"]
    branch = values["interpretation"]
    language = {
        "inversion": "identify label inversion",
        "neither_failure": "do not identify label inversion",
    }[branch]
    return rf"""\paragraph{{Prespecified construct check.}}
In the prespecified \ConstructCell{{}} follow-up, orientation calibration changed median fixed-evaluator damage from \ConstructRawMedian{{}} to \ConstructOrientedMedian{{}} across \ConstructCandidates{{}} candidate edits. Fresh linear decoders reached median final-test accuracy \ConstructFreshLinear{{}} on edited states versus their \ConstructFreshLinearBaseline{{}} unedited baseline; fresh MLP decoders reached \ConstructFreshMLP{{}} versus \ConstructFreshMLPBaseline{{}}. No materiality threshold or inferential recovery rule was predeclared, so these fresh-decoder comparisons remain descriptive. The calibration-frozen orientation endpoints {language} in this cell; they do not establish erasure elsewhere. Orientation was selected only on the disjoint calibration split and frozen for final-test evaluation.

\begin{{figure}}[t]
  \centering
  \includegraphics[width=\linewidth]{{figures/fig_orientation_redecodability.pdf}}
  \caption{{Construct checks for the 60 prespecified candidate-conditioned edits in Qwen2.5-1.5B/SST-2/layer 14. Panels align raw fixed-evaluator damage, calibration-frozen orientation damage, and fresh linear/MLP target-label accuracy. Horizontal marks denote medians; every point is one candidate edit.}}
  \label{{fig:orientation-redecodability}}
\end{{figure}}"""


def _validate_paper_targets(text: str) -> None:
    for name in ("EPSILON-SWEEP", "ORIENTATION-REDECODABILITY"):
        begin = f"% BEGIN POST-RUN {name} UPDATE"
        end = f"% END POST-RUN {name} UPDATE"
        if text.count(begin) != 1 or text.count(end) != 1:
            raise ManuscriptValidationError(f"required {name} marker pair is missing or duplicated")
        if text.index(begin) > text.index(end):
            raise ManuscriptValidationError(f"required {name} marker order is invalid")
    if "figures/fig_circularity.pdf" not in text:
        raise ManuscriptValidationError("expanded figure reference target is missing")
    macro_pattern = re.compile(
        r"^\\newcommand\{\\SplitCells\}\{.*?^\\newcommand\{\\SplitP\}\{[^\n]*\}\n",
        flags=re.MULTILINE | re.DOTALL,
    )
    if len(macro_pattern.findall(text)) != 1:
        raise ManuscriptValidationError("numeric macro region is missing or duplicated")


def patch_manuscript(
    source_path: str | Path,
    destination_path: str | Path,
    analysis_summary: str | Path | Mapping[str, Any],
    *,
    macros_path: str | Path | None = None,
) -> PaperPatchReport:
    """Patch only locked markers plus enumerated macro, scope, and figure targets."""

    summary = validate_analysis_summary(analysis_summary)
    source = Path(source_path)
    destination = Path(destination_path)
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ManuscriptValidationError(f"could not read manuscript source: {exc}") from exc
    _validate_paper_targets(text)

    macro_text = "\n".join(_macro_lines(summary)) + "\n"
    macro_pattern = re.compile(
        r"^\\newcommand\{\\SplitCells\}\{.*?^\\newcommand\{\\SplitP\}\{[^\n]*\}\n",
        flags=re.MULTILINE | re.DOTALL,
    )
    text, substitutions = macro_pattern.subn(lambda _match: macro_text, text, count=1)
    if substitutions != 1:
        raise ManuscriptValidationError("numeric macro patch did not apply exactly once")

    cells = int(summary["matched_split"]["included_units"]["cells"])
    blocks = int(summary["matched_split"]["included_units"]["model_task_blocks"])
    pairs = int(summary["matched_split"]["included_units"]["pairs"])
    analyzed_cells = int(summary["matched_split"]["analyzed_units"]["cells"])
    analyzed_pairs = int(summary["matched_split"]["analyzed_units"]["pairs"])
    score_null_pairs = int(summary["matched_split"]["primary_score_null_pairs"])
    score_null_cells = int(summary["matched_split"]["primary_score_null_cells"])
    depths = cells // blocks
    matched_floor_note = ""
    matched_result_note = ""
    if score_null_pairs:
        matched_floor_note = (
            f" The locked denominator floor yielded {score_null_pairs} affected "
            f"pair units; all {pairs} requested pairs remain in the row artifact, "
            f"while both conditions of each affected pair are excluded symmetrically, "
            f"leaving {analyzed_pairs} pairs and {analyzed_cells} analyzable cells in "
            f"the available-case estimand. {score_null_cells} requested cell is "
            "explicitly non-estimable."
        )
        matched_result_note = (
            f" The estimate uses {analyzed_pairs} complete pairs; the "
            f"{score_null_pairs} denominator-floor pairs remain explicit in the row "
            "artifact with both conditions excluded symmetrically."
        )
    text = _replace_exact(
        text,
        r"over \SplitCells{} model--task cells",
        f"over {cells} requested model--layer--task cells ({analyzed_cells} "
        f"analyzable) nested in {blocks} model--task blocks",
        "abstract scope",
    )
    text = _replace_exact(
        text,
        r"gap \SplitGap{}, 95\% cell-bootstrap interval \SplitCI{}; exact sign-flip $p=\SplitP$",
        r"gap \SplitGap{}, 95\% hierarchical block-bootstrap interval \SplitCI{}; exact block sign-flip $p=\SplitP$",
        "abstract inferential wording",
    )
    text = _replace_exact(
        text,
        r"a gap of \SplitGap{} across \SplitCells{} model--task cells.",
        f"an available-case gap of \\SplitGap{{}} across {analyzed_cells} "
        f"analyzable cells from {cells} requested model--layer--task cells nested "
        f"in {blocks} model--task blocks.",
        "introduction scope",
    )
    text = _replace_exact(
        text,
        r"The 95\% cell-bootstrap interval is \SplitCI{}, and an exact two-sided sign-flip test gives $p=\SplitP$.",
        r"The 95\% hierarchical block-bootstrap interval is \SplitCI{}, and the primary exact two-sided block sign-flip test gives $p=\SplitP$.",
        "introduction inference",
    )
    old_design = (
        "The matched-versus-split control holds the edit fixed and changes only the evaluator. "
        "For each of six models, we use the middle layer on SVA and SST-2. Five independently "
        "seeded attacker/evaluator pairs train on sentence-disjoint examples. The attacker "
        "generates each edit. The edited representations are then scored either by that attacker "
        "(matched) or by its paired evaluator (split). We average the five pair-level contrasts "
        "within each model--task cell and conduct inference over the \\SplitCells{} cell means."
    )
    new_design = (
        "The matched-versus-split control holds each edited tensor fixed and changes only the "
        f"evaluator. For each of six models, we use {depths} prespecified sampled layers on SVA "
        "and SST-2. Five independently seeded attacker/evaluator pairs train on sentence-disjoint "
        "examples. The attacker generates each edit, and both conditions score its identical "
        "tensor. We average the five pair contrasts within each model--layer--task cell and use "
        f"the {blocks} model--task blocks as the independent units for primary inference "
        f"({pairs} requested pair units total).{matched_floor_note}"
    )
    text = _replace_exact(text, old_design, new_design, "matched/split design paragraph")
    old_result = (
        "AlterRep's mean target damage drops from \\SplitMatched{} to \\SplitIndependent{}. "
        "The mean paired gap is \\SplitGap{}, with a 95\\% cell-bootstrap interval of "
        "\\SplitCI{} and exact two-sided sign-flip $p=\\SplitP$. All seven nonzero cell gaps "
        "are positive; the other five are zero. The effect varies across model--task cells. "
        "Because the edited representations are held fixed, the gap identifies evaluator reuse "
        "as a source of measured damage in this pipeline."
    )
    new_result = (
        f"Across {analyzed_cells} analyzable, equally weighted model--layer--task cells "
        f"from {cells} requested cells, AlterRep mean target "
        "damage drops from \\SplitMatched{} to \\SplitIndependent{}. The mean paired gap is "
        "\\SplitGap{}, with a 95\\% hierarchical block-bootstrap interval of \\SplitCI{} and "
        "primary exact two-sided block sign-flip $p=\\SplitP$. Because both conditions score "
        "the identical edited tensor, the gap identifies evaluator reuse as a source of measured "
        f"damage in this pipeline.{matched_result_note}"
    )
    text = _replace_exact(text, old_result, new_result, "matched/split result paragraph")

    text = _replace_marker(text, "EPSILON-SWEEP", _epsilon_prose(summary))

    old_figure = r"\includegraphics[width=0.92\linewidth]{figures/fig_circularity.pdf}"
    new_figure = r"\includegraphics[width=\linewidth]{figures/fig_circularity_expanded.pdf}"
    text = _replace_exact(text, old_figure, new_figure, "expanded figure path")
    old_caption = (
        "\\caption{Matched-versus-split scoring across \\SplitCells{} model--task cells. "
        "AlterRep damage falls when the split evaluator scores the attacker's edit. FGSM and "
        "PGD are saturated at the configured perturbation budget. Horizontal marks show grand means.}"
    )
    new_caption = (
        f"\\caption{{Matched-versus-split AlterRep target damage for {analyzed_cells} "
        f"analyzable cells from {cells} requested model--layer--task cells nested in "
        f"{blocks} model--task blocks. Panel A shows "
        "paired cell means for the identical edits; the grand matched and split means are "
        "\\SplitMatched{} and \\SplitIndependent{}. Panel B shows all block depth trajectories "
        "and the equal-block mean gap with a 95\\% hierarchical interval. Denominator-floor "
        f"nulls affect {score_null_pairs} of {pairs} requested pair units and are excluded "
        "symmetrically.}"
    )
    text = _replace_exact(text, old_caption, new_caption, "expanded figure caption")

    text = _replace_marker(
        text,
        "ORIENTATION-REDECODABILITY",
        _construct_prose(summary),
    )

    old_scope = (
        "The original archives also lack Phase-2 per-example predictions, edited representations "
        "for fresh-decoder training, failed layer-evaluator accuracies, and Hessian eigenvectors. "
        "Orientation and post-edit redecodability are therefore untested in the archived study. "
        "We include them in Table~\\ref{tab:protocol} because they are necessary falsification "
        "checks, not because our pipeline satisfied them."
    )
    new_scope = (
        "The original archives still lack Phase-2 per-example predictions, edited representations "
        "for fresh-decoder training, failed layer-evaluator accuracies, and Hessian eigenvectors. "
        "Orientation and fresh-decoder recovery were tested in one prespecified model--layer--task "
        "cell in the revision follow-up; the original archive and the remaining cells still lack "
        "these endpoints. The one-cell result is a falsification check, not evidence of erasure "
        "elsewhere."
    )
    text = _replace_exact(text, old_scope, new_scope, "construct scope paragraph")

    old_appendix = (
        "For each of six models, we use the middle layer for SVA and SST-2. Five seeded "
        "attacker/evaluator pairs train on separate examples. The attacker generates each edit. "
        "Target and control damage are then scored once by that attacker and once by its paired "
        "evaluator. We compute the primary AlterRep contrast on the \\SplitCells{} cell means. "
        "The reported percentile interval resamples cells with replacement; the exact sign-flip "
        "test flips the sign of each nonzero paired cell difference. A second bootstrap that "
        "resamples both cells and the five within-cell pairs gives $[0.164,0.529]$."
    )
    new_appendix = (
        f"For each of six models, we use {depths} prespecified sampled layers for SVA and SST-2. "
        "Five seeded attacker/evaluator pairs train on separate sentence groups. Each attacker "
        "generates one edit that is hashed and scored unchanged by its matched and split "
        f"evaluators, yielding {pairs} pair units in {cells} cells nested in {blocks} model--task "
        f"blocks. The locked denominator floor excludes both conditions for {score_null_pairs} "
        f"affected pairs from numerical summaries ({analyzed_pairs} analyzed pairs and "
        f"{analyzed_cells} analyzable cells) while "
        "retaining every requested row and reason in the artifact. We average valid pairs "
        "within cells and layers within equally weighted blocks. The "
        "primary 95\\% interval hierarchically resamples model--task blocks, layers, and pairs; "
        "the primary exact sign-flip test flips the nonzero block contrasts."
    )
    text = _replace_exact(text, old_appendix, new_appendix, "appendix matched/split scope")

    macro_destination = (
        Path(macros_path) if macros_path is not None else destination.parent / "manuscript_numbers.tex"
    )
    _atomic_text(
        macro_destination,
        "% Generated from validated analysis_summary.json; do not edit.\n" + macro_text,
    )
    _atomic_text(destination, text)
    return PaperPatchReport(
        destination=destination,
        macros_path=macro_destination,
        changed_regions=(
            "numeric_macros",
            "matched_split_scope",
            "epsilon_marker",
            "expanded_figure",
            "orientation_marker",
            "construct_scope",
            "appendix_scope",
        ),
    )


def _extension_number(
    record: Mapping[str, Any],
    *names: str,
    location: str,
) -> float:
    for name in names:
        value = record.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                return number
    raise ManuscriptValidationError(
        f"{location} requires one finite numeric field from {list(names)!r}"
    )


def _extension_integer(
    record: Mapping[str, Any],
    *names: str,
    location: str,
    minimum: int = 0,
) -> int:
    for name in names:
        value = record.get(name)
        if type(value) is int and value >= minimum:
            return value
    raise ManuscriptValidationError(
        f"{location} requires one integer field from {list(names)!r}"
    )


def _extension_cell_key(value: Any, *, location: str) -> tuple[str, str, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or not isinstance(value[0], str)
        or not value[0]
        or not isinstance(value[1], str)
        or not value[1]
        or type(value[2]) is not int
    ):
        raise ManuscriptValidationError(
            f"{location} must be a [model_key, task, layer] cell key"
        )
    return str(value[0]), str(value[1]), int(value[2])


def _validate_exact_sign_flip(
    record: Mapping[str, Any],
    sign_counts: Mapping[str, Any],
    *,
    expected_mean: float,
    location: str,
) -> None:
    if (
        record.get("method") != "two_sided_exact_paired_sign_flip"
        or record.get("status") != "ok"
        or record.get("zero_tolerance") != 0.0
        or record.get("n_blocks") != 12
    ):
        raise ManuscriptValidationError(
            f"{location} exact sign-flip metadata is invalid"
        )
    nonzero = _extension_integer(record, "nonzero_count", location=location)
    zero = _extension_integer(record, "zero_count", location=location)
    permutations = _extension_integer(record, "permutations", location=location, minimum=1)
    extreme = _extension_integer(record, "extreme_count", location=location, minimum=1)
    p_value = _extension_number(record, "p_value", location=location)
    observed_mean = _extension_number(record, "observed_mean", location=location)
    if (
        nonzero + zero != 12
        or permutations != 1 << nonzero
        or extreme > permutations
        or not math.isclose(p_value, extreme / permutations, abs_tol=1.0e-15)
        or not math.isclose(observed_mean, expected_mean, abs_tol=1.0e-12)
    ):
        raise ManuscriptValidationError(
            f"{location} exact sign-flip arithmetic is inconsistent"
        )
    positive = _extension_integer(sign_counts, "positive", location=f"{location}.sign_counts")
    sign_zero = _extension_integer(sign_counts, "zero", location=f"{location}.sign_counts")
    negative = _extension_integer(sign_counts, "negative", location=f"{location}.sign_counts")
    if (
        positive + sign_zero + negative != 12
        or positive + negative != nonzero
        or sign_zero != zero
    ):
        raise ManuscriptValidationError(
            f"{location} exact sign-flip block signs are inconsistent"
        )


def _validate_raw_bootstrap(record: Mapping[str, Any], *, expected_gap: float) -> None:
    required = {
        "method": "equal_block_hierarchical_percentile_bootstrap",
        "cluster_unit": "model_key_task",
        "hierarchy": ["model_task", "layer", "pair"],
        "estimand_weighting": "equal_model_task_blocks",
        "draws": 10_000,
        "seed": 20260830,
        "confidence": 0.95,
        "n_blocks": 12,
        "n_cells": 60,
        "n_pairs": 300,
    }
    if any(record.get(key) != value for key, value in required.items()):
        raise ManuscriptValidationError(
            "full-case raw hierarchical bootstrap metadata differs from the locked analysis"
        )
    for key in ("point_estimate", "equal_block_point_estimate"):
        if not math.isclose(
            _extension_number(record, key, location="full_case_raw_drop.bootstrap"),
            expected_gap,
            abs_tol=1.0e-12,
        ):
            raise ManuscriptValidationError(
                "full-case raw hierarchical bootstrap point estimate is inconsistent"
            )


def validate_extension_summary(
    source: str | Path | Mapping[str, Any],
) -> None:
    """Fail closed before extension figures or prose consume scientific claims."""

    summary = _load_summary(source)
    robustness = _require_mapping(summary, "floor_robustness", "extension")
    if robustness.get("schema") != "reviewer_revision.floor_robustness.v1":
        raise ManuscriptValidationError("floor robustness schema is invalid")
    if robustness.get("schema_version") != 1 or robustness.get("status") != "ok":
        raise ManuscriptValidationError("floor robustness must be complete and status ok")
    if (
        robustness.get("label") != "post_hoc_sensitivity"
        or robustness.get("post_hoc_sensitivity") is not True
        or robustness.get("replaces_registered_primary") is not False
    ):
        raise ManuscriptValidationError(
            "floor robustness must remain a post-hoc sensitivity and not replace the primary"
        )
    units = _require_mapping(robustness, "units", "floor_robustness")
    expected_units = {
        "rows": 600,
        "pairs": 300,
        "cells": 60,
        "model_task_blocks": 12,
    }
    if any(units.get(key) != value for key, value in expected_units.items()):
        raise ManuscriptValidationError(
            "floor robustness must retain all 600 rows, 300 pairs, 60 cells, and 12 blocks"
        )
    raw = _require_mapping(
        robustness, "full_case_raw_drop", "floor_robustness"
    )
    if (
        raw.get("label") != "post_hoc_sensitivity"
        or raw.get("post_hoc_sensitivity") is not True
        or raw.get("replaces_registered_primary") is not False
    ):
        raise ManuscriptValidationError(
            "full-case raw drop must remain post-hoc and must not replace the primary"
        )
    if (
        raw.get("pairs") != 300
        or raw.get("cells") != 60
        or raw.get("model_task_blocks") != 12
    ):
        raise ManuscriptValidationError(
            "full-case raw drop must retain 300 pairs, 60 cells, and 12 blocks"
        )
    matched = _extension_number(raw, "matched_mean", "matched", location="full_case_raw_drop")
    split = _extension_number(raw, "split_mean", "split", location="full_case_raw_drop")
    gap = _extension_number(raw, "gap", location="full_case_raw_drop")
    if not math.isclose(matched - split, gap, abs_tol=1.0e-12):
        raise ManuscriptValidationError("full-case raw-drop means do not reproduce its gap")
    raw_bootstrap = _require_mapping(raw, "bootstrap", "full_case_raw_drop")
    _validate_raw_bootstrap(raw_bootstrap, expected_gap=gap)
    ci_low = _extension_number(raw_bootstrap, "ci_low", location="full_case_raw_drop.bootstrap")
    ci_high = _extension_number(raw_bootstrap, "ci_high", location="full_case_raw_drop.bootstrap")
    if ci_low > ci_high:
        raise ManuscriptValidationError("full-case raw-drop interval is reversed")
    raw_sign = _require_mapping(raw, "exact_sign_flip", "full_case_raw_drop")
    raw_sign_counts = _require_mapping(
        raw, "block_gap_sign_counts", "full_case_raw_drop"
    )
    _validate_exact_sign_flip(
        raw_sign,
        raw_sign_counts,
        expected_mean=gap,
        location="full-case raw drop",
    )

    curve = robustness.get("floor_curve")
    if not isinstance(curve, list) or len(curve) != len(LOCKED_FLOOR_GRID):
        raise ManuscriptValidationError("floor curve must contain the exact five locked floors")
    observed_floors: list[float] = []
    previous_pairs = 301
    previous_cells = 61
    locked_floor_record: dict[str, Any] | None = None
    planned_cell_universe: set[tuple[str, str, int]] | None = None
    for index, row in enumerate(curve):
        if not isinstance(row, Mapping):
            raise ManuscriptValidationError("floor curve rows must be mappings")
        row = dict(row)
        floor = _extension_number(row, "floor", location=f"floor_curve[{index}]")
        observed_floors.append(floor)
        if row.get("label") != "post_hoc_sensitivity":
            raise ManuscriptValidationError(
                f"floor_curve[{index}] must remain post-hoc sensitivity"
            )
        if row.get("requested_pairs") != 300:
            raise ManuscriptValidationError(
                f"floor_curve[{index}] requested pair count must remain 300"
            )
        analyzed_pairs = _extension_integer(
            row,
            "analyzed_pairs",
            location=f"floor_curve[{index}]",
            minimum=1,
        )
        if row.get("pairs") != analyzed_pairs or analyzed_pairs > 300:
            raise ManuscriptValidationError(
                f"floor_curve[{index}] pair counts are inconsistent"
            )
        excluded_keys = row.get("excluded_pair_keys")
        excluded_rows = row.get("excluded_pairs")
        if (
            not isinstance(excluded_keys, list)
            or not isinstance(excluded_rows, list)
            or len(excluded_keys) != 300 - analyzed_pairs
            or len(excluded_rows) != 300 - analyzed_pairs
        ):
            raise ManuscriptValidationError(
                f"floor_curve[{index}] pair counts disagree with exclusions"
            )
        if row.get("requested_cells") != 60:
            raise ManuscriptValidationError(
                f"floor_curve[{index}] requested cell count must remain 60"
            )
        analyzed_cells = _extension_integer(
            row, "analyzed_cells", location=f"floor_curve[{index}]", minimum=1
        )
        full_cells = _extension_integer(
            row, "full_cells", location=f"floor_curve[{index}]"
        )
        partial_cells = _extension_integer(
            row, "partial_cells", location=f"floor_curve[{index}]"
        )
        missing_cells = _extension_integer(
            row, "missing_cells", location=f"floor_curve[{index}]"
        )
        if (
            full_cells + partial_cells + missing_cells != 60
            or full_cells + partial_cells != analyzed_cells
        ):
            raise ManuscriptValidationError(
                f"floor_curve[{index}] cell coverage counts are inconsistent"
            )
        coverage_sets: list[set[tuple[str, str, int]]] = []
        for key_name, expected_count in (
            ("full_cell_keys", full_cells),
            ("partial_cell_keys", partial_cells),
            ("missing_cell_keys", missing_cells),
        ):
            values = row.get(key_name)
            if not isinstance(values, list) or len(values) != expected_count:
                raise ManuscriptValidationError(
                    f"floor_curve[{index}] {key_name} disagrees with its count"
                )
            parsed = {
                _extension_cell_key(
                    value,
                    location=f"floor_curve[{index}].{key_name}",
                )
                for value in values
            }
            if len(parsed) != expected_count:
                raise ManuscriptValidationError(
                    f"floor_curve[{index}] {key_name} contains duplicate cells"
                )
            coverage_sets.append(parsed)
        full_set, partial_set, missing_set = coverage_sets
        if (
            full_set.intersection(partial_set)
            or full_set.intersection(missing_set)
            or partial_set.intersection(missing_set)
        ):
            raise ManuscriptValidationError(
                f"floor_curve[{index}] cell coverage categories overlap"
            )
        row_universe = full_set | partial_set | missing_set
        if len(row_universe) != 60:
            raise ManuscriptValidationError(
                f"floor_curve[{index}] does not enumerate all 60 planned cells"
            )
        if planned_cell_universe is None:
            planned_cell_universe = row_universe
        elif row_universe != planned_cell_universe:
            raise ManuscriptValidationError(
                "floor-curve planned cell identities differ across floors"
            )
        if (
            row.get("requested_model_task_blocks") != 12
            or row.get("analyzed_model_task_blocks") != 12
        ):
            raise ManuscriptValidationError(
                f"floor_curve[{index}] must retain all 12 analyzed model-task blocks"
            )
        if analyzed_pairs > previous_pairs or analyzed_cells > previous_cells:
            raise ManuscriptValidationError(
                "floor-curve coverage must be monotone nonincreasing as the floor rises"
            )
        previous_pairs = analyzed_pairs
        previous_cells = analyzed_cells
        matched_mean = _extension_number(
            row, "matched_mean", location=f"floor_curve[{index}]"
        )
        split_mean = _extension_number(
            row, "split_mean", location=f"floor_curve[{index}]"
        )
        gap = _extension_number(row, "gap", location=f"floor_curve[{index}]")
        if not math.isclose(matched_mean - split_mean, gap, abs_tol=1.0e-12):
            raise ManuscriptValidationError(
                f"floor_curve[{index}] matched-minus-split arithmetic is inconsistent"
            )
        equal_block = _require_mapping(row, "equal_block", f"floor_curve[{index}]")
        for key, expected in (
            ("matched", matched_mean),
            ("split", split_mean),
            ("gap", gap),
        ):
            observed = _extension_number(
                equal_block, key, location=f"floor_curve[{index}].equal_block"
            )
            if not math.isclose(observed, expected, abs_tol=1.0e-12):
                raise ManuscriptValidationError(
                    f"floor_curve[{index}] equal-block aliases are inconsistent"
                )
        sign = _require_mapping(row, "exact_sign_flip", f"floor_curve[{index}]")
        sign_counts = _require_mapping(
            row, "block_gap_sign_counts", f"floor_curve[{index}]"
        )
        _validate_exact_sign_flip(
            sign,
            sign_counts,
            expected_mean=gap,
            location=f"floor_curve[{index}]",
        )
        if math.isclose(floor, 0.55, abs_tol=1.0e-12):
            locked_floor_record = row
    if any(
        not math.isclose(observed, expected, abs_tol=1.0e-12)
        for observed, expected in zip(observed_floors, LOCKED_FLOOR_GRID, strict=True)
    ):
        raise ManuscriptValidationError("floor curve differs from the exact locked grid")
    if curve[0].get("analyzed_pairs") != 300:
        raise ManuscriptValidationError(
            "the floor just above chance must retain all 300 planned pairs"
        )
    if locked_floor_record is None:
        raise ManuscriptValidationError("floor curve lacks the locked floor 0.55")
    partial = _require_mapping(
        robustness, "partial_identification", "floor_robustness"
    )
    if (
        partial.get("label") != "deterministic_partial_identification_bound"
        or partial.get("sensitivity_label") != "post_hoc_sensitivity"
        or partial.get("is_confidence_interval") is not False
    ):
        raise ManuscriptValidationError(
            "partial-identification output must be a deterministic bound, not a confidence interval"
        )
    lower = _extension_number(partial, "lower", location="partial_identification")
    upper = _extension_number(partial, "upper", location="partial_identification")
    if lower > upper:
        raise ManuscriptValidationError("partial-identification bound is reversed")
    if not math.isclose(
        _extension_number(partial, "floor", location="partial_identification"),
        0.55,
        abs_tol=1.0e-12,
    ):
        raise ManuscriptValidationError(
            "partial-identification analysis must use the locked floor 0.55"
        )
    locked_gap = _extension_number(
        locked_floor_record, "gap", location="floor_curve.locked_floor"
    )
    available_gap = _extension_number(
        partial, "available_case_gap", location="partial_identification"
    )
    if not math.isclose(available_gap, locked_gap, abs_tol=1.0e-12):
        raise ManuscriptValidationError(
            "partial-identification available-case gap differs from the locked curve"
        )
    if partial.get("missing_pair_gap_domain") != [-1.0, 1.0]:
        raise ManuscriptValidationError(
            "partial-identification missing-gap domain must remain [-1, 1]"
        )
    planned_pairs = _extension_integer(
        partial, "planned_pairs", location="partial_identification", minimum=1
    )
    observed_pairs = _extension_integer(
        partial, "observed_pairs", location="partial_identification", minimum=1
    )
    missing_pairs = _extension_integer(
        partial, "missing_pairs", location="partial_identification"
    )
    if (
        planned_pairs != 300
        or observed_pairs != locked_floor_record["analyzed_pairs"]
        or observed_pairs + missing_pairs != planned_pairs
        or partial.get("planned_cells") != 60
        or partial.get("fixed_model_task_blocks") != 12
    ):
        raise ManuscriptValidationError(
            "partial-identification planned, observed, and missing counts are inconsistent"
        )
    wholly_missing = _extension_integer(
        partial,
        "wholly_missing_cells",
        location="partial_identification",
    )
    if wholly_missing != locked_floor_record["missing_cells"]:
        raise ManuscriptValidationError(
            "partial-identification wholly-missing cell count is inconsistent"
        )
    cell_bounds = partial.get("cell_bounds")
    if not isinstance(cell_bounds, list) or len(cell_bounds) != 60:
        raise ManuscriptValidationError(
            "partial-identification must retain bounds for all 60 planned cells"
        )
    if planned_cell_universe is None:
        raise ManuscriptValidationError(
            "partial-identification lacks a validated planned-cell universe"
        )
    observed_total = 0
    wholly_missing_total = 0
    observed_keys: set[tuple[str, str, int]] = set()
    bound_categories: dict[str, set[tuple[str, str, int]]] = {
        "full": set(),
        "partial": set(),
        "missing": set(),
    }
    block_records: dict[tuple[str, str], list[tuple[float, float, float | None]]] = {}
    for index, raw_record in enumerate(cell_bounds):
        if not isinstance(raw_record, Mapping):
            raise ManuscriptValidationError(
                "partial-identification cell bounds must all be mappings"
            )
        record = dict(raw_record)
        key = _extension_cell_key(
            [record.get("model_key"), record.get("task"), record.get("layer")],
            location=f"partial_identification.cell_bounds[{index}]",
        )
        if key in observed_keys:
            raise ManuscriptValidationError(
                "partial-identification cell bounds must identify 60 unique planned cells"
            )
        observed_keys.add(key)
        planned = _extension_integer(
            record,
            "planned_pairs",
            location=f"partial_identification.cell_bounds[{index}]",
            minimum=1,
        )
        observed = _extension_integer(
            record,
            "observed_pairs",
            location=f"partial_identification.cell_bounds[{index}]",
        )
        missing = _extension_integer(
            record,
            "missing_pairs",
            location=f"partial_identification.cell_bounds[{index}]",
        )
        if planned != 5 or observed > planned or observed + missing != planned:
            raise ManuscriptValidationError(
                "partial-identification cell planned, observed, and missing counts are inconsistent"
            )
        cell_lower = _extension_number(
            record,
            "lower",
            location=f"partial_identification.cell_bounds[{index}]",
        )
        cell_upper = _extension_number(
            record,
            "upper",
            location=f"partial_identification.cell_bounds[{index}]",
        )
        expected_width = 2.0 * missing / planned
        if (
            not -1.0 <= cell_lower <= cell_upper <= 1.0
            or not math.isclose(
                cell_upper - cell_lower,
                expected_width,
                abs_tol=1.0e-12,
            )
        ):
            raise ManuscriptValidationError(
                "partial-identification cell bound width is inconsistent with missing pairs"
            )
        observed_sum = 0.5 * (cell_lower + cell_upper) * planned
        if abs(observed_sum) > observed + 1.0e-12:
            raise ManuscriptValidationError(
                "partial-identification cell bound center is outside the paired-gap domain"
            )
        observed_mean: float | None
        if observed == 0:
            wholly_missing_total += 1
            observed_mean = None
            category = "missing"
            if not (
                math.isclose(cell_lower, -1.0, abs_tol=1.0e-12)
                and math.isclose(cell_upper, 1.0, abs_tol=1.0e-12)
            ):
                raise ManuscriptValidationError(
                    "a wholly missing cell must retain the full [-1, 1] bound"
                )
        else:
            observed_mean = observed_sum / observed
            category = "full" if missing == 0 else "partial"
        bound_categories[category].add(key)
        observed_total += observed
        block_records.setdefault((key[0], key[1]), []).append(
            (cell_lower, cell_upper, observed_mean)
        )
    if observed_keys != planned_cell_universe:
        raise ManuscriptValidationError(
            "partial-identification cell bounds differ from the 60 unique planned cells"
        )
    if observed_total != observed_pairs:
        raise ManuscriptValidationError(
            "partial-identification cell bounds do not reproduce observed pairs"
        )
    if wholly_missing_total != wholly_missing:
        raise ManuscriptValidationError(
            "partial-identification cell bounds disagree on wholly missing cells"
        )
    locked_coverage = {
        category: {
            _extension_cell_key(
                value,
                location=f"floor_curve.locked_floor.{category}_cell_keys",
            )
            for value in locked_floor_record[f"{category}_cell_keys"]
        }
        for category in ("full", "partial", "missing")
    }
    if any(
        bound_categories[category] != locked_coverage[category]
        for category in ("full", "partial", "missing")
    ):
        raise ManuscriptValidationError(
            "partial-identification cell coverage differs from the locked floor"
        )
    if len(block_records) != 12 or any(
        len(records) != 5 for records in block_records.values()
    ):
        raise ManuscriptValidationError(
            "partial-identification must retain five planned cells in each of 12 blocks"
        )
    block_lowers = [
        math.fsum(record[0] for record in records) / len(records)
        for records in block_records.values()
    ]
    block_uppers = [
        math.fsum(record[1] for record in records) / len(records)
        for records in block_records.values()
    ]
    recomputed_lower = math.fsum(block_lowers) / len(block_lowers)
    recomputed_upper = math.fsum(block_uppers) / len(block_uppers)
    if not (
        math.isclose(lower, recomputed_lower, abs_tol=1.0e-12)
        and math.isclose(upper, recomputed_upper, abs_tol=1.0e-12)
    ):
        raise ManuscriptValidationError(
            "partial-identification reported endpoints differ from the recomputed equal-block bound"
        )
    block_available_means: list[float] = []
    for records in block_records.values():
        estimable = [record[2] for record in records if record[2] is not None]
        if not estimable:
            raise ManuscriptValidationError(
                "partial-identification leaves a model-task block wholly unobserved"
            )
        block_available_means.append(
            math.fsum(float(value) for value in estimable) / len(estimable)
        )
    recomputed_available = math.fsum(block_available_means) / len(
        block_available_means
    )
    if not math.isclose(available_gap, recomputed_available, abs_tol=1.0e-12):
        raise ManuscriptValidationError(
            "partial-identification cell bounds do not reproduce the available-case gap"
        )
    if not -1.0 <= lower <= available_gap <= upper <= 1.0:
        raise ManuscriptValidationError(
            "partial-identification bound must contain the locked available-case gap"
        )

    construct = _require_mapping(summary, "construct_panel", "extension")
    if construct.get("schema") != "reviewer_revision.construct_panel_inference.v1":
        raise ManuscriptValidationError("construct panel schema is invalid")
    if construct.get("schema_version") != 1 or construct.get("status") != "ok":
        raise ManuscriptValidationError("construct panel must be complete and status ok")
    bootstrap_settings = _require_mapping(
        construct, "bootstrap", "construct_panel"
    )
    required_bootstrap = {
        "draws": 10_000,
        "seed": 20260830,
        "bit_generator": "PCG64",
        "resampling_unit": "final_test_group_id",
        "shared_resamples_across_models_within_task": True,
    }
    if any(
        bootstrap_settings.get(key) != value
        for key, value in required_bootstrap.items()
    ):
        raise ManuscriptValidationError(
            "construct panel bootstrap settings differ from the locked group bootstrap"
        )
    if (
        construct.get("endpoint_tail_p_value")
        != "(1 + count(theta_star <= threshold)) / (B + 1)"
        or construct.get("within_cell_combination")
        != "intersection_union_max_endpoint_p"
    ):
        raise ManuscriptValidationError(
            "construct panel endpoint tail or intersection-union rule is invalid"
        )
    if construct.get("confirmatory_cell_count") != 11:
        raise ManuscriptValidationError("construct panel must contain exactly 11 confirmatory cells")
    cells = _require_mapping(construct, "cells", "construct_panel")
    if set(cells) != LOCKED_EXTENSION_CELLS:
        raise ManuscriptValidationError("construct panel must contain the exact 12 locked cells")
    pilot = _require_mapping(construct, "pilot", "construct_panel")
    if pilot != cells[LOCKED_EXTENSION_PILOT]:
        raise ManuscriptValidationError("construct pilot record differs from its locked cell")
    if pilot.get("confirmatory") is not False or pilot.get("inference_mode") != "descriptive_only":
        raise ManuscriptValidationError("construct pilot must remain descriptive and non-confirmatory")
    forbidden_pilot = sorted(PILOT_FORBIDDEN_INFERENCE_FIELDS.intersection(pilot))
    if forbidden_pilot:
        raise ManuscriptValidationError(
            f"construct pilot contains confirmatory inferential fields: {forbidden_pilot}"
        )

    nonestimable = 0
    raw_cell_p_values: dict[str, float] = {}
    expected_point_pass: dict[str, bool] = {}
    confirmatory_records: dict[str, dict[str, Any]] = {}
    for slug, value in cells.items():
        if not isinstance(value, Mapping):
            raise ManuscriptValidationError(f"construct cell {slug} must be a mapping")
        record = dict(value)
        identity = _require_mapping(record, "cell", f"construct_panel.cells.{slug}")
        expected_slug = f"{identity.get('model_key')}-{identity.get('task')}-l{identity.get('layer')}"
        if expected_slug != slug:
            raise ManuscriptValidationError(f"construct cell identity does not match slug {slug}")
        confirmatory = slug != LOCKED_EXTENSION_PILOT
        if record.get("confirmatory") is not confirmatory:
            raise ManuscriptValidationError(f"construct confirmatory flag is wrong for {slug}")
        status = record.get("status")
        if status not in {"ok", "nonestimable"}:
            raise ManuscriptValidationError(f"construct cell {slug} has invalid status")
        if status == "nonestimable":
            if confirmatory:
                nonestimable += 1
                if (
                    record.get("internal_cell_p_value") != 1.0
                    or record.get("passes_locked_point_thresholds") is not False
                    or record.get("passes_locked_confirmatory_rule") is not False
                ):
                    raise ManuscriptValidationError(
                        f"nonestimable construct cell {slug} must remain in-family with p=1"
                    )
                raw_cell_p_values[slug] = 1.0
                expected_point_pass[slug] = False
                confirmatory_records[slug] = record
            continue
        if record.get("n_candidate_edits") != 60:
            raise ManuscriptValidationError(
                f"construct cell {slug} must contain exactly 60 candidate edits"
            )
        endpoints = _require_mapping(record, "endpoints", f"construct_panel.cells.{slug}")
        endpoint_values = {
            name: _extension_number(
                endpoints,
                endpoint,
                location=f"construct_panel.cells.{slug}.endpoints",
            )
            for name, endpoint in CONSTRUCT_ENDPOINT_FIELDS.items()
        }
        if record.get("thresholds") != LOCKED_CONSTRUCT_THRESHOLDS:
            raise ManuscriptValidationError(
                f"construct cell {slug} differs from the locked point-threshold rules"
            )
        expected_decisions = {
            name: endpoint_values[name] >= LOCKED_CONSTRUCT_THRESHOLDS[name]
            for name in LOCKED_CONSTRUCT_THRESHOLDS
        }
        if record.get("point_threshold_decisions") != expected_decisions:
            raise ManuscriptValidationError(
                f"construct cell {slug} point-threshold decisions do not reproduce its endpoints"
            )
        expected_points = all(expected_decisions.values())
        if record.get("passes_locked_point_thresholds") is not expected_points:
            raise ManuscriptValidationError(
                f"construct cell {slug} point-threshold pass flag is inconsistent"
            )
        if not confirmatory:
            continue
        if record.get("within_cell_combination") != construct.get(
            "within_cell_combination"
        ):
            raise ManuscriptValidationError(
                f"construct cell {slug} intersection-union rule is inconsistent"
            )
        bounds = _require_mapping(
            record, "marginal_lower_bounds", f"construct_panel.cells.{slug}"
        )
        bound_finite = _require_mapping(
            record,
            "marginal_lower_bound_finite",
            f"construct_panel.cells.{slug}",
        )
        if set(bounds) != set(LOCKED_CONSTRUCT_THRESHOLDS) or set(
            bound_finite
        ) != set(LOCKED_CONSTRUCT_THRESHOLDS):
            raise ManuscriptValidationError(
                f"construct cell {slug} marginal-bound endpoint keys are invalid"
            )
        for endpoint in LOCKED_CONSTRUCT_THRESHOLDS:
            finite = bound_finite[endpoint]
            if type(finite) is not bool:
                raise ManuscriptValidationError(
                    f"construct cell {slug} marginal-bound finite flag is not Boolean"
                )
            if finite:
                _extension_number(
                    bounds,
                    endpoint,
                    location=f"construct_panel.cells.{slug}.marginal_lower_bounds",
                )
            elif bounds[endpoint] is not None:
                raise ManuscriptValidationError(
                    f"construct cell {slug} nonfinite marginal bound must be null"
                )
        if (
            record.get("lower_bound_scope") != "marginal"
            or record.get("lower_bound_multiplicity_adjusted") is not False
            or record.get("lower_bound_simultaneous") is not False
        ):
            raise ManuscriptValidationError(
                f"construct cell {slug} mislabels its marginal lower bounds"
            )
        for key in (
            "passes_locked_point_thresholds",
            "passes_holm_adjusted_inference",
            "passes_locked_confirmatory_rule",
        ):
            if type(record.get(key)) is not bool:
                raise ManuscriptValidationError(f"construct cell {slug} lacks Boolean {key}")
        endpoint_p_values = _require_mapping(
            record, "endpoint_p_values", f"construct_panel.cells.{slug}"
        )
        if set(endpoint_p_values) != set(LOCKED_CONSTRUCT_THRESHOLDS):
            raise ManuscriptValidationError(
                f"construct cell {slug} endpoint p-value keys are invalid"
            )
        checked_endpoint_p = {
            endpoint: _extension_number(
                endpoint_p_values,
                endpoint,
                location=f"construct_panel.cells.{slug}.endpoint_p_values",
            )
            for endpoint in LOCKED_CONSTRUCT_THRESHOLDS
        }
        if any(not 0.0 <= value <= 1.0 for value in checked_endpoint_p.values()):
            raise ManuscriptValidationError(
                f"construct cell {slug} endpoint p-value is outside [0, 1]"
            )
        if any(
            not math.isclose(value * 10_001, round(value * 10_001), abs_tol=1.0e-10)
            or value < 1 / 10_001
            for value in checked_endpoint_p.values()
        ):
            raise ManuscriptValidationError(
                f"construct cell {slug} endpoint p-value does not follow the plus-one 10,000-draw rule"
            )
        internal_p = _extension_number(
            record,
            "internal_cell_p_value",
            location=f"construct_panel.cells.{slug}",
        )
        if not math.isclose(
            internal_p, max(checked_endpoint_p.values()), abs_tol=1.0e-15
        ):
            raise ManuscriptValidationError(
                f"construct cell {slug} intersection-union p-value is inconsistent"
            )
        raw_cell_p_values[slug] = internal_p
        expected_point_pass[slug] = expected_points
        confirmatory_records[slug] = record
    if set(raw_cell_p_values) != LOCKED_EXTENSION_CELLS - {LOCKED_EXTENSION_PILOT}:
        raise ManuscriptValidationError(
            "construct panel did not preserve the exact 11-cell Holm family"
        )
    ordered_p = sorted(raw_cell_p_values.items(), key=lambda item: (item[1], item[0]))
    running_adjusted = 0.0
    expected_adjusted: dict[str, float] = {}
    for rank, (slug, value) in enumerate(ordered_p):
        candidate = min(1.0, (len(ordered_p) - rank) * value)
        running_adjusted = max(running_adjusted, candidate)
        expected_adjusted[slug] = running_adjusted
    passing = 0
    point_passing = 0
    holm_passing = 0
    for slug, record in confirmatory_records.items():
        reported_adjusted = _extension_number(
            record,
            "holm_adjusted_cell_p_value",
            location=f"construct_panel.cells.{slug}",
        )
        if not math.isclose(
            reported_adjusted, expected_adjusted[slug], abs_tol=1.0e-15
        ):
            raise ManuscriptValidationError(
                f"construct cell {slug} Holm-adjusted p-value is inconsistent"
            )
        expected_inference = expected_adjusted[slug] <= 0.05
        if record.get("passes_holm_adjusted_inference") is not expected_inference:
            raise ManuscriptValidationError(
                f"construct cell {slug} Holm decision is inconsistent"
            )
        expected_combined = expected_point_pass[slug] and expected_inference
        if record.get("passes_locked_confirmatory_rule") is not expected_combined:
            raise ManuscriptValidationError(
                f"construct cell {slug} combined confirmatory decision is inconsistent"
            )
        point_passing += int(expected_point_pass[slug])
        holm_passing += int(expected_inference)
        passing += int(expected_combined)
    if construct.get("passing_confirmatory_cell_count") != passing:
        raise ManuscriptValidationError("construct passing-cell count disagrees with cell records")
    if construct.get("nonestimable_confirmatory_cell_count") != nonestimable:
        raise ManuscriptValidationError("construct nonestimable-cell count disagrees with cell records")
    multiplicity = _require_mapping(construct, "multiplicity", "construct_panel")
    if multiplicity != {"method": "holm_one_sided", "family_size": 11, "alpha": 0.05}:
        raise ManuscriptValidationError("construct panel must use one-sided Holm over 11 cells")
    lower_bounds = _require_mapping(construct, "lower_bounds", "construct_panel")
    if (
        lower_bounds.get("confidence_level") != 0.95
        or lower_bounds.get("quantile") != 0.05
        or lower_bounds.get("quantile_method") != "linear"
        or lower_bounds.get("scope") != "marginal"
        or lower_bounds.get("multiplicity_adjusted") is not False
        or lower_bounds.get("simultaneous") is not False
    ):
        raise ManuscriptValidationError(
            "construct lower bounds must be marginal 95% bounds, neither adjusted nor simultaneous"
        )
    construct["_validated_point_passing_count"] = point_passing
    construct["_validated_holm_passing_count"] = holm_passing


def _extension_macro_text(summary: Mapping[str, Any]) -> str:
    robustness = dict(summary["floor_robustness"])
    raw = dict(robustness["full_case_raw_drop"])
    raw_bootstrap = dict(raw["bootstrap"])
    raw_sign = dict(raw["exact_sign_flip"])
    locked = next(
        dict(row)
        for row in robustness["floor_curve"]
        if math.isclose(float(row["floor"]), 0.55, abs_tol=1.0e-12)
    )
    partial = dict(robustness["partial_identification"])
    construct = dict(summary["construct_panel"])
    confirmatory_records = [
        dict(record)
        for record in dict(construct["cells"]).values()
        if record.get("confirmatory") is True
    ]
    point_passing = sum(
        record.get("passes_locked_point_thresholds") is True
        for record in confirmatory_records
    )
    holm_passing = sum(
        record.get("passes_holm_adjusted_inference") is True
        for record in confirmatory_records
    )
    lines = [
        "% Generated from validated extension_summary.json; do not edit.",
        rf"\newcommand{{\FloorRawMatched}}{{{_extension_number(raw, 'matched_mean', 'matched', location='full_case_raw_drop'):.3f}}}",
        rf"\newcommand{{\FloorRawSplit}}{{{_extension_number(raw, 'split_mean', 'split', location='full_case_raw_drop'):.3f}}}",
        rf"\newcommand{{\FloorRawGap}}{{{_extension_number(raw, 'gap', location='full_case_raw_drop'):.3f}}}",
        rf"\newcommand{{\FloorRawCI}}{{${'['}{_extension_number(raw_bootstrap, 'ci_low', location='full_case_raw_drop.bootstrap'):.3f}, {_extension_number(raw_bootstrap, 'ci_high', location='full_case_raw_drop.bootstrap'):.3f}{']'}$}}",
        rf"\newcommand{{\FloorRawP}}{{{_extension_number(raw_sign, 'p_value', location='full_case_raw_drop.exact_sign_flip'):.6f}}}",
        rf"\newcommand{{\FloorLockedPairs}}{{{_extension_integer(locked, 'analyzed_pairs', 'pairs', location='locked_floor', minimum=1)}}}",
        rf"\newcommand{{\FloorLockedGap}}{{{_extension_number(locked, 'gap', location='locked_floor'):.3f}}}",
        rf"\newcommand{{\FloorPartialBound}}{{${'['}{_extension_number(partial, 'lower', location='partial_identification'):.3f}, {_extension_number(partial, 'upper', location='partial_identification'):.3f}{']'}$}}",
        r"\newcommand{\ConstructConfirmatoryCells}{11}",
        rf"\newcommand{{\ConstructPointPassingCells}}{{{point_passing}}}",
        rf"\newcommand{{\ConstructHolmPassingCells}}{{{holm_passing}}}",
        rf"\newcommand{{\ConstructPassingCells}}{{{int(construct['passing_confirmatory_cell_count'])}}}",
        rf"\newcommand{{\ConstructNonestimableCells}}{{{int(construct['nonestimable_confirmatory_cell_count'])}}}",
    ]
    return "\n".join(lines) + "\n"


def _extension_floor_prose() -> str:
    return r"""\paragraph{Denominator-floor robustness.}
As a post-hoc full-case sensitivity, the raw target-accuracy-drop contrast uses all 300 paired edits: matched and split mean drops are \FloorRawMatched{} and \FloorRawSplit{}, for an equal-block gap of \FloorRawGap{} (95\% hierarchical interval \FloorRawCI{}; exact block sign-flip $p=\FloorRawP$). This check has no unstable chance-normalized denominator and does not replace the registered normalized-damage primary analysis. Across the complete prespecified floor curve, the locked $0.55$ floor retains \FloorLockedPairs{} pairs and gives gap \FloorLockedGap{}. Bounding each missing paired normalized-damage gap in $[-1,1]$ yields the deterministic partial-identification bound \FloorPartialBound{}; it is not a confidence interval.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.78\linewidth]{figures/fig_floor_sensitivity.pdf}
  \caption{Post-hoc denominator-floor sensitivity. Panel A shows the equal-block matched-minus-split contrast over every prespecified floor, the full-case raw-drop contrast, and the locked-floor deterministic partial-identification bound. Panel B shows analyzed-pair coverage. The registered available-case analysis remains primary.}
  \label{fig:floor-sensitivity}
\end{figure}"""


def _extension_construct_prose() -> str:
    return r"""\paragraph{Prospective construct panel.}
The inspected Qwen/SST-2/layer-14 result remains a disclosed descriptive pilot. We prospectively evaluated the same fresh-linear recovery endpoints in 11 untouched middle-layer cells using shared task-group resamples across models. Of those cells, \ConstructPointPassingCells{} pass all three locked point thresholds, \ConstructHolmPassingCells{} pass the one-sided Holm-adjusted cell rule, and \ConstructPassingCells{} pass both; \ConstructNonestimableCells{} are nonestimable and remain in the fixed 11-cell family with internal $p=1$. The displayed one-sided 95\% lower bounds are marginal, neither multiplicity-adjusted nor simultaneous; Holm-adjusted decisions are reported separately. These task-decoder endpoints are not downstream language-model behavior, and a passing decoder battery does not prove concept erasure. The pilot's calibration-frozen orientation, AlterRep, and fresh-MLP compatibility endpoints remain reported in Appendix~\ref{app:pilot-compatibility}.

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/fig_construct_panel.pdf}
  \caption{Pilot-plus-confirmatory construct panel. Points summarize the fixed 60 dependent candidate edits in each cell; horizontal segments are marginal one-sided 95\% lower bounds, with edge arrows for nonfinite conservative bounds. Dashed lines mark the locked accuracy, target-recovery, and control-retention thresholds. The diamond is the descriptive pilot and is excluded from Holm adjustment; N/E marks a nonestimable cell without assigning it an endpoint value.}
  \label{fig:construct-panel}
\end{figure}"""


def _extension_construct_context_prose() -> str:
    return (
        "We ran the candidate-identity, evaluator-separation, saturation, "
        "data-overlap, and dependence-aware inference checks. The registered "
        "archive lacked the per-example artifacts required for retrospective "
        "orientation and fresh-decoder tests. A disclosed descriptive pilot and "
        "the separately registered 11-cell prospective panel provide those "
        "follow-up checks without changing the original analysis."
    )


def _extension_construct_scope_prose() -> str:
    return (
        "The original archives still lack Phase-2 per-example predictions, edited "
        "representations for fresh-decoder training, failed layer-evaluator "
        "accuracies, and Hessian eigenvectors. The new extension artifacts support "
        "the disclosed pilot and prospective middle-layer construct panel, but do "
        "not repair the historical archive or turn task-decoder recovery into "
        "evidence about downstream language-model behavior. The finite decoder "
        "battery remains a falsification check and does not prove concept erasure."
    )


def patch_manuscript_with_extension(
    source_path: str | Path,
    destination_path: str | Path,
    summary_source: str | Path | Mapping[str, Any],
    *,
    macros_path: str | Path | None = None,
) -> PaperPatchReport:
    """Patch only the four allowlisted extension regions after strict validation."""

    validate_extension_summary(summary_source)
    summary = _load_summary(summary_source)
    source = Path(source_path)
    destination = Path(destination_path)
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ManuscriptValidationError(f"could not load manuscript source: {exc}") from exc
    text = _replace_marker(
        text,
        "CONSTRUCT-CONTEXT",
        _extension_construct_context_prose(),
    )
    text = _replace_marker(text, "FLOOR-SENSITIVITY", _extension_floor_prose())
    text = _replace_marker(text, "CONSTRUCT-PANEL", _extension_construct_prose())
    text = _replace_marker(
        text,
        "CONSTRUCT-SCOPE",
        _extension_construct_scope_prose(),
    )
    macro_destination = (
        Path(macros_path)
        if macros_path is not None
        else destination.parent / "extension_numbers.tex"
    )
    if macro_destination.parent.resolve() != destination.parent.resolve():
        raise ManuscriptValidationError(
            "extension macros must be written beside the patched manuscript"
        )
    macro_input = rf"\input{{{macro_destination.name}}}"
    if macro_input not in text:
        text = _replace_exact(
            text,
            r"\begin{document}",
            r"\begin{document}" + "\n" + macro_input,
            "extension macro input",
        )
    _atomic_text(macro_destination, _extension_macro_text(summary))
    _atomic_text(destination, text)
    return PaperPatchReport(
        destination=destination,
        macros_path=macro_destination,
        changed_regions=(
            "construct_context",
            "floor_sensitivity",
            "construct_panel",
            "construct_scope",
            "extension_numeric_macros",
        ),
    )


__all__ = [
    "ManuscriptValidationError",
    "PaperPatchReport",
    "generate_manuscript_numbers",
    "patch_manuscript",
    "patch_manuscript_with_extension",
    "validate_analysis_summary",
    "validate_extension_summary",
]
