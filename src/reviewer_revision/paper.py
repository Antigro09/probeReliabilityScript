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


__all__ = [
    "ManuscriptValidationError",
    "PaperPatchReport",
    "generate_manuscript_numbers",
    "patch_manuscript",
    "validate_analysis_summary",
]
