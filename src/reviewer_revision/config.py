"""Validation and exact key expansion for the locked reviewer experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple

import yaml

from .artifacts import canonical_json, sha256_json


class ConfigError(ValueError):
    """Raised when the revision specification differs from its locked design."""


class MatchedSplitCellKey(NamedTuple):
    model_key: str
    task: str
    layer: int


class MatchedSplitRowKey(NamedTuple):
    model_key: str
    task: str
    layer: int
    pair_seed: int
    method: str
    condition: str


class EpsilonSweepRowKey(NamedTuple):
    model_key: str
    task: str
    layer: int
    pair_seed: int
    method: str
    epsilon: float
    condition: str


class ConstructEditKey(NamedTuple):
    edit_kind: str
    architecture: str
    seed: int

    @property
    def edit_id(self) -> str:
        return f"{self.edit_kind}:{self.architecture}:seed-{self.seed}"


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConfigError(f"unhashable YAML mapping key: {key!r}") from exc
        if duplicate:
            raise ConfigError(f"duplicate YAML mapping key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


_LOCKED_MODEL_KEYS = ("pythia", "gpt2", "bert", "qwen", "gemma", "llama")
_LOCKED_TASKS = ("sva", "sst2")
_LOCKED_PAIR_SEEDS = (0, 1, 2, 3, 4)
_LOCKED_EPSILONS = (
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
_COUPLED_METHODS = ("alterrep", "fgsm", "pgd")
_REFERENCE_METHODS = ("inlp", "rlace")
_SCORING_CONDITIONS = ("matched", "split")

_LOCKED_REPRODUCIBILITY = {
    "master_seed": 20260829,
    "pair_seeds": list(_LOCKED_PAIR_SEEDS),
    "bootstrap_seed": 20260830,
    "bootstrap_draws": 10_000,
    "permutation_seed": 20260831,
    "float_dtype_for_probe_training": "float32",
    "deterministic_algorithms": True,
    "preserve_existing_artifacts": True,
    "overwrite": False,
}

_LOCKED_RUNTIME = {
    "device_priority": ["mps", "cpu"],
    "transformer_extraction_device": "mps",
    "linear_probe_device": "cpu",
    "statistics_device": "cpu",
    "env": {
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "TOKENIZERS_PARALLELISM": "false",
    },
    "full_grid_max_projected_hours": 48,
    "epsilon_all_layers_max_projected_hours": 12,
    "maximum_projected_disk_gb": 80,
}

_LOCKED_MODELS = [
    {
        "key": "pythia",
        "hf_id": "EleutherAI/pythia-160m",
        "layers": [1, 4, 6, 9, 12],
        "middle_layer": 6,
    },
    {
        "key": "gpt2",
        "hf_id": "openai-community/gpt2",
        "layers": [1, 4, 6, 9, 12],
        "middle_layer": 6,
    },
    {
        "key": "bert",
        "hf_id": "google-bert/bert-base-uncased",
        "layers": [1, 4, 6, 9, 12],
        "middle_layer": 6,
    },
    {
        "key": "qwen",
        "hf_id": "Qwen/Qwen2.5-1.5B",
        "layers": [1, 8, 14, 21, 28],
        "middle_layer": 14,
    },
    {
        "key": "gemma",
        "hf_id": "google/gemma-2-2b",
        "layers": [1, 7, 14, 20, 26],
        "middle_layer": 14,
    },
    {
        "key": "llama",
        "hf_id": "meta-llama/Llama-3.2-3B",
        "layers": [1, 8, 14, 21, 28],
        "middle_layer": 14,
    },
]

_LOCKED_SCORE = {
    "chance": 0.5,
    "denominator_floor": 0.55,
    "target_damage": (
        "clip((target_acc_pre - target_acc_post) / (target_acc_pre - 0.5), 0, 1)"
    ),
    "control_preservation": (
        "clip((control_acc_post - 0.5) / (control_acc_pre - 0.5), 0, 1)"
    ),
    "harmonic_mean": "2 * C * S / (C + S), with H(0,0)=0",
}

_LOCKED_BASELINE_INTERVENTIONS = {
    "alterrep": {"alpha": 1.0},
    "fgsm": {"norm": "linf", "epsilon": 0.5},
    "pgd": {
        "norm": "linf",
        "epsilon": 0.5,
        "steps": 10,
        "step_size": 0.1,
        "step_size_rule_for_sweep": "epsilon / 5",
    },
}

_LOCKED_EXPERIMENT_A = {
    "primary_method": "alterrep",
    "secondary_methods": ["fgsm", "pgd"],
    "reference_controls": ["inlp", "rlace"],
    "scoring_conditions": ["matched", "split"],
    "full_grid": {
        "models": "all",
        "tasks": ["sva", "sst2"],
        "layer_policy": "all_five_sampled_layers",
        "expected_cells": 60,
        "expected_model_task_blocks": 12,
        "expected_pair_units": 300,
    },
    "fallback_grid": {
        "trigger": "projected_full_grid_hours_above_runtime_limit",
        "models": "all",
        "tasks": ["sva", "sst2"],
        "layer_policy": "first_middle_last",
        "expected_cells": 36,
        "expected_model_task_blocks": 12,
        "expected_pair_units": 180,
    },
    "hard_floor": {
        "minimum_total_cells": 36,
        "rule": "do_not_reduce_below_this_without_a_structured_fatal_blocker",
    },
    "estimand": (
        "For each model-layer-task-pair, Delta = C_matched - C_split for the "
        "exact same attacker-generated edited representations. Average pair "
        "deltas within cell, then average cells with equal weight."
    ),
    "inference": {
        "primary_cluster": "model_task",
        "primary_exact_test": "two_sided_sign_flip_over_12_model_task_block_means",
        "primary_interval": "hierarchical_percentile_bootstrap",
        "bootstrap_hierarchy": [
            "model_task_block",
            "layer_within_block",
            "pair_within_cell",
        ],
        "secondary_test": "two_sided_sign_flip_over_model_layer_task_cell_means",
        "secondary_label": "sensitivity_only_layers_share_examples",
        "report": [
            "grand_mean_matched",
            "grand_mean_split",
            "grand_mean_gap",
            "median_gap",
            "fraction_cell_gaps_positive",
            "fraction_block_gaps_positive",
            "depth_position_means",
            "task_specific_means",
            "leave_one_model_out_means",
            "all_zero_and_nonzero_counts",
        ],
    },
}

_LOCKED_EXPERIMENT_B = {
    "methods": ["fgsm", "pgd"],
    "scoring_conditions": ["matched", "split"],
    "required_scope": {
        "models": "all",
        "tasks": ["sva", "sst2"],
        "layers": "middle_only",
        "expected_cells": 12,
    },
    "automatic_extension": {
        "condition": "projected_all_layer_sweep_hours_at_or_below_runtime_limit",
        "scope": "use_experiment_a_selected_cells",
    },
    "epsilons": list(_LOCKED_EPSILONS),
    "pgd": {
        "steps": 10,
        "step_size_rule": "epsilon / 5",
        "preserve_random_start_behavior_from_baseline_implementation": True,
    },
    "paired_randomness": {
        "reuse_attacker_evaluator_weights_across_epsilons": True,
        "reuse_example_order_across_epsilons": True,
        "reuse_pgd_random_start_base_noise_across_epsilons": True,
    },
    "integrity": {
        "epsilon_zero_is_exact_noop": True,
        "epsilon_zero_max_abs_representation_delta_tolerance": 1.0e-6,
        "epsilon_zero_target_damage_tolerance": 1.0e-6,
        "enforce_linf_bound": True,
        "baseline_epsilon_must_reproduce_archived_pattern": True,
    },
    "report": [
        "raw_target_accuracy_pre_and_post",
        "raw_control_accuracy_pre_and_post",
        "C",
        "S",
        "H",
        "orientation_sensitivity_accuracy",
        "auc",
        "log_loss",
        "ceiling_indicator",
        "realized_linf_norm",
    ],
    "figure": {
        "x_axis": "epsilon_log2_with_zero_shown_separately",
        "y_axis": "mean_target_damage_C",
        "panels": ["fgsm", "pgd"],
        "lines": ["matched", "split"],
        "uncertainty": "model_task_cluster_bootstrap_95_percent",
        "companion_panel_or_table": "fraction_at_C_equal_1_by_epsilon",
    },
}

_LOCKED_CONSTRUCT = {
    "cell": {
        "model_key": "qwen",
        "model_id": "Qwen/Qwen2.5-1.5B",
        "task": "sst2",
        "layer": 14,
        "selection_rule": (
            "Largest archived AlterRep matched-minus-split gap among non-Gemma, "
            "high-sample SST-2 middle-layer cells; selected before the new run."
        ),
    },
    "candidate_scope": {
        "architectures": ["linear", "mlp", "mka"],
        "seeds": list(range(20)),
        "expected_candidates": 60,
    },
    "edit_objects": {
        "required": ["alterrep", "dcand_crossfit"],
        "epsilon_methods_for_orientation_only": ["fgsm", "pgd"],
    },
    "source_split": {
        "preserve_phase2_candidate_fraction": 0.40,
        "preserve_phase2_evaluator_fraction": 0.20,
        "subdivide_phase2_intervention_fraction": {
            "direction_fit": "one_third",
            "fresh_decoder_fit": "one_third",
            "orientation_calibration": "one_sixth",
            "final_test": "one_sixth",
        },
        "preserve_phase2_test_fraction_unused_for_this_check": 0.10,
        "grouping_unit": "sentence_id_or_existing_dedup_group",
        "stratify_by": "joint_target_control_label",
    },
    "direction_estimation": {
        "dcand_crossfit_uses_only_direction_fit": True,
        "scoring_never_uses_direction_fit_examples": True,
        "normalize_direction_l2": True,
    },
    "fixed_evaluator_metrics": {
        "raw": [
            "accuracy",
            "balanced_accuracy",
            "confusion_matrix",
            "auc",
            "log_loss",
        ],
        "orientation": {
            "choose_identity_or_flip_on": "orientation_calibration",
            "apply_choice_to": "final_test",
            "report": [
                "calibrated_orientation_accuracy",
                "max_accuracy_sensitivity",
                "orientation_adjusted_auc",
                "orientation_adjusted_log_loss",
                "C_raw",
                "C_orientation_calibrated",
            ],
        },
    },
    "fresh_decoders": {
        "train_on": "edited_fresh_decoder_fit",
        "evaluate_on": "edited_final_test",
        "tune_only_within": "fresh_decoder_fit",
        "hyperparameter_selection": (
            "Select once on unedited fresh_decoder_fit data for each decoder "
            "family and label using an inner group-disjoint validation split; "
            "freeze and reuse the selected settings for every edit."
        ),
        "target_and_control": True,
        "unedited_baseline": True,
        "required_models": {
            "linear": {
                "implementation": "reuse_repository_linear_probe",
                "learning_rates": [0.0003, 0.001, 0.003],
                "weight_decays": [0.0, 0.001, 0.01, 0.1],
                "epochs": 50,
                "batch_size": 256,
                "early_stopping_patience": 5,
            },
            "mlp": {
                "implementation": "reuse_repository_mlp_probe",
                "hidden_dim": 256,
                "learning_rate": 0.001,
                "weight_decay": 0.01,
                "epochs": 50,
                "batch_size": 256,
                "early_stopping_patience": 5,
                "seeds": [0, 1, 2],
            },
        },
        "derived_quantities": {
            "target_recovery_ratio": "(post_edit_acc - 0.5) / (unedited_acc - 0.5)",
            "control_retention_ratio": (
                "(post_edit_control_acc - 0.5) / (unedited_control_acc - 0.5)"
            ),
        },
    },
    "save": [
        "split_manifest_with_example_ids_and_hashes",
        "candidate_checkpoint_hashes",
        "evaluator_checkpoint_hashes",
        "direction_vectors",
        "per_example_pre_and_post_logits",
        "per_example_predictions",
        "raw_and_orientation_adjusted_confusion_matrices",
        "edited_representation_shards_or_lossless_reconstruction_recipe",
        "fresh_decoder_checkpoints",
        "hyperparameter_selection_records",
    ],
    "interpretation_rules": {
        "inversion": "raw_post_accuracy_below_0_5_but_calibrated_orientation_accuracy_high",
        "redecodable": "fresh_decoder_post_accuracy_materially_above_0_5",
        "limitation": "absence_of_recovery_in_one_cell_is_not_proof_of_erasure",
    },
}

_LOCKED_OUTPUTS = {
    "root": "results/reviewer_revision_2026_08",
    "immutable_run_directory": True,
    "required_files": [
        "run_manifest.json",
        "environment.json",
        "preflight_report.json",
        "runtime_benchmark.json",
        "matched_split_rows.parquet",
        "matched_split_rows.csv",
        "matched_split_summary.json",
        "epsilon_sweep_rows.parquet",
        "epsilon_sweep_rows.csv",
        "epsilon_sweep_summary.json",
        "construct_check_rows.parquet",
        "construct_check_rows.csv",
        "construct_check_summary.json",
        "analysis_summary.json",
        "manuscript_numbers.tex",
        "validation_report.md",
        "console.log",
    ],
    "figures": {
        "directory": "figures",
        "formats": ["pdf", "png"],
        "png_dpi": 300,
        "names": {
            "expanded_matched_split": "fig_circularity_expanded",
            "epsilon_sweep": "fig_epsilon_sweep",
            "construct_check": "fig_orientation_redecodability",
        },
    },
}

_LOCKED_PAPER_UPDATE = {
    "source": "main_revised.tex",
    "bibliography": "references.bib",
    "update_numbers_from": "results/reviewer_revision_2026_08/manuscript_numbers.tex",
    "required_markers": [
        "BEGIN POST-RUN EPSILON-SWEEP UPDATE",
        "BEGIN POST-RUN ORIENTATION-REDECODABILITY UPDATE",
    ],
    "terminology": {
        "preferred_condition_names": ["matched", "split"],
        "never_equate_fixed_decoder_damage_with_erasure": True,
        "distinguish_cells_from_independent_model_task_blocks": True,
    },
    "compile": {
        "engine": "latexmk_pdf",
        "fail_on_undefined_references": True,
        "fail_on_missing_citations": True,
        "fail_on_overfull_hbox_over_3pt": True,
        "render_and_visually_inspect": True,
    },
}


def _mapping(parent: dict[str, Any], key: str, location: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{location}.{key} must be a mapping")
    return value


def _sequence(parent: dict[str, Any], key: str, location: str) -> list[Any]:
    value = parent.get(key)
    if not isinstance(value, list):
        raise ConfigError(f"{location}.{key} must be a list")
    return value


def _require_equal(actual: Any, expected: Any, location: str) -> None:
    boolean_type_mismatch = isinstance(actual, bool) != isinstance(expected, bool)
    if boolean_type_mismatch or actual != expected:
        raise ConfigError(f"{location} must be {expected!r}, got {actual!r}")


def _require_locked_subset(actual: Any, expected: Any, location: str) -> None:
    """Recursively validate locked semantic values with path-specific errors.

    Mapping order does not affect validation. Unknown keys inside a locked
    section are rejected so a misspelled or unsupported control cannot be
    silently ignored. List order remains significant where it controls
    deterministic execution order.
    """

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ConfigError(f"{location} must be a mapping")
        for key in actual:
            if key not in expected:
                child_location = f"{location}.{key}" if location else str(key)
                raise ConfigError(f"{child_location} is not a recognized locked parameter")
        for key, expected_value in expected.items():
            child_location = f"{location}.{key}" if location else str(key)
            if key not in actual:
                raise ConfigError(f"{child_location} is required")
            _require_locked_subset(actual[key], expected_value, child_location)
        return
    if isinstance(expected, list):
        if not isinstance(actual, list):
            raise ConfigError(f"{location} must be a list")
        if len(actual) != len(expected):
            raise ConfigError(
                f"{location} must contain {len(expected)} entries, got {len(actual)}"
            )
        for index, (actual_value, expected_value) in enumerate(zip(actual, expected)):
            _require_locked_subset(
                actual_value, expected_value, f"{location}[{index}]"
            )
        return
    _require_equal(actual, expected, location)


@dataclass(frozen=True)
class RevisionConfig:
    """Validated revision specification plus deterministic key expansion."""

    raw: dict[str, Any]
    source_path: Path
    canonical_json: str
    config_hash: str

    @property
    def models(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.raw["models"])

    @property
    def tasks(self) -> tuple[str, ...]:
        return tuple(self.raw["tasks"]["required"])

    @property
    def pair_seeds(self) -> tuple[int, ...]:
        return tuple(self.raw["reproducibility"]["pair_seeds"])

    @property
    def epsilons(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.raw["experiment_b_epsilon_sweep"]["epsilons"])

    def model(self, model_key: str) -> dict[str, Any]:
        for model in self.models:
            if model["key"] == model_key:
                return model
        raise KeyError(model_key)

    def matched_split_cells(
        self, grid: Literal["full", "fallback"] = "full"
    ) -> tuple[MatchedSplitCellKey, ...]:
        if grid not in {"full", "fallback"}:
            raise ConfigError(f"unknown matched/split grid: {grid!r}")
        cells: list[MatchedSplitCellKey] = []
        for model in self.models:
            layers = list(model["layers"])
            if grid == "fallback":
                layers = [layers[0], model["middle_layer"], layers[-1]]
            for task in self.tasks:
                for layer in layers:
                    cells.append(MatchedSplitCellKey(model["key"], task, int(layer)))
        return tuple(cells)

    def matched_split_row_keys(
        self, grid: Literal["full", "fallback"] = "full"
    ) -> tuple[MatchedSplitRowKey, ...]:
        rows: list[MatchedSplitRowKey] = []
        for cell in self.matched_split_cells(grid):
            for pair_seed in self.pair_seeds:
                for method in _COUPLED_METHODS:
                    for condition in _SCORING_CONDITIONS:
                        rows.append(MatchedSplitRowKey(*cell, pair_seed, method, condition))
                for method in _REFERENCE_METHODS:
                    rows.append(MatchedSplitRowKey(*cell, pair_seed, method, "reference"))
        return tuple(rows)

    def epsilon_sweep_row_keys(self) -> tuple[EpsilonSweepRowKey, ...]:
        rows: list[EpsilonSweepRowKey] = []
        for model in self.models:
            layer = int(model["middle_layer"])
            for task in self.tasks:
                for pair_seed in self.pair_seeds:
                    for method in self.raw["experiment_b_epsilon_sweep"]["methods"]:
                        for epsilon in self.epsilons:
                            for condition in self.raw["experiment_b_epsilon_sweep"][
                                "scoring_conditions"
                            ]:
                                rows.append(
                                    EpsilonSweepRowKey(
                                        model["key"],
                                        task,
                                        layer,
                                        pair_seed,
                                        method,
                                        epsilon,
                                        condition,
                                    )
                                )
        return tuple(rows)

    def construct_edit_keys(self) -> tuple[ConstructEditKey, ...]:
        candidate = self.raw["experiment_c_orientation_redecodability"]["candidate_scope"]
        keys = [ConstructEditKey("alterrep", "attacker", seed) for seed in self.pair_seeds]
        keys.extend(
            ConstructEditKey("dcand_crossfit", architecture, int(seed))
            for architecture in candidate["architectures"]
            for seed in candidate["seeds"]
        )
        return tuple(keys)


def _validate_config(raw: dict[str, Any]) -> None:
    _require_equal(raw.get("schema_version"), 1, "schema_version")
    _require_equal(raw.get("run_name"), "reviewer_revision_2026_08", "run_name")

    reproducibility = _mapping(raw, "reproducibility", "root")
    _require_locked_subset(
        reproducibility, _LOCKED_REPRODUCIBILITY, "reproducibility"
    )
    _require_equal(
        tuple(_sequence(reproducibility, "pair_seeds", "reproducibility")),
        _LOCKED_PAIR_SEEDS,
        "reproducibility.pair_seeds",
    )
    for key in ("master_seed", "bootstrap_seed", "bootstrap_draws", "permutation_seed"):
        if not isinstance(reproducibility.get(key), int):
            raise ConfigError(f"reproducibility.{key} must be an integer")
    _require_equal(reproducibility.get("bootstrap_draws"), 10_000, "reproducibility.bootstrap_draws")
    _require_equal(reproducibility.get("overwrite"), False, "reproducibility.overwrite")

    runtime = _mapping(raw, "runtime", "root")
    _require_locked_subset(runtime, _LOCKED_RUNTIME, "runtime")
    _require_equal(runtime.get("device_priority"), ["mps", "cpu"], "runtime.device_priority")
    _require_equal(runtime.get("full_grid_max_projected_hours"), 48, "runtime.full_grid_max_projected_hours")
    _require_equal(runtime.get("maximum_projected_disk_gb"), 80, "runtime.maximum_projected_disk_gb")

    models = _sequence(raw, "models", "root")
    model_keys = [model.get("key") if isinstance(model, dict) else None for model in models]
    _require_equal(tuple(model_keys), _LOCKED_MODEL_KEYS, "models keys")
    if len(model_keys) != len(set(model_keys)):
        raise ConfigError("duplicate model key")
    for index, model in enumerate(models):
        if not isinstance(model, dict):
            raise ConfigError(f"models[{index}] must be a mapping")
        layers = _sequence(model, "layers", f"models[{index}]")
        if len(layers) != 5 or any(not isinstance(layer, int) for layer in layers):
            raise ConfigError(f"models[{index}].layers must contain five integer layers")
        if len(layers) != len(set(layers)):
            raise ConfigError(f"duplicate layer in models[{index}].layers")
        if layers != sorted(layers):
            raise ConfigError(f"models[{index}].layers must be strictly increasing")
        if model.get("middle_layer") not in layers:
            raise ConfigError(f"models[{index}].middle_layer must be sampled")
        if not isinstance(model.get("hf_id"), str) or not model["hf_id"]:
            raise ConfigError(f"models[{index}].hf_id must be nonempty")
    _require_locked_subset(models, _LOCKED_MODELS, "models")

    tasks = _mapping(raw, "tasks", "root")
    _require_locked_subset(
        tasks,
        {
            "required": ["sva", "sst2"],
            "excluded": ["gender"],
            "exclusion_reason": (
                "WinoGender is not part of the reviewer-requested expansion; "
                "the archived version was underpowered and the original Phase-1 "
                "preprocessing was broken."
            ),
        },
        "tasks",
    )
    _require_equal(tuple(_sequence(tasks, "required", "tasks")), _LOCKED_TASKS, "tasks.required")
    if set(tasks.get("excluded", [])) & set(_LOCKED_TASKS):
        raise ConfigError("required and excluded tasks overlap")

    score = _mapping(raw, "shared_score_definition", "root")
    _require_locked_subset(score, _LOCKED_SCORE, "shared_score_definition")
    _require_equal(score.get("chance"), 0.5, "shared_score_definition.chance")
    _require_equal(score.get("denominator_floor"), 0.55, "shared_score_definition.denominator_floor")

    baseline = _mapping(raw, "baseline_interventions", "root")
    _require_locked_subset(
        baseline, _LOCKED_BASELINE_INTERVENTIONS, "baseline_interventions"
    )
    _require_equal(_mapping(baseline, "alterrep", "baseline_interventions").get("alpha"), 1.0, "baseline_interventions.alterrep.alpha")
    for method in ("fgsm", "pgd"):
        section = _mapping(baseline, method, "baseline_interventions")
        _require_equal(section.get("epsilon"), 0.5, f"baseline_interventions.{method}.epsilon")
    pgd_baseline = _mapping(baseline, "pgd", "baseline_interventions")
    _require_equal(pgd_baseline.get("steps"), 10, "baseline_interventions.pgd.steps")

    experiment_a = _mapping(raw, "experiment_a_expanded_matched_split", "root")
    _require_locked_subset(
        experiment_a,
        _LOCKED_EXPERIMENT_A,
        "experiment_a_expanded_matched_split",
    )
    _require_equal(experiment_a.get("primary_method"), "alterrep", "experiment_a.primary_method")
    _require_equal(tuple(experiment_a.get("secondary_methods", [])), ("fgsm", "pgd"), "experiment_a.secondary_methods")
    _require_equal(tuple(experiment_a.get("reference_controls", [])), _REFERENCE_METHODS, "experiment_a.reference_controls")
    _require_equal(tuple(experiment_a.get("scoring_conditions", [])), _SCORING_CONDITIONS, "experiment_a.scoring_conditions")
    full = _mapping(experiment_a, "full_grid", "experiment_a")
    fallback = _mapping(experiment_a, "fallback_grid", "experiment_a")
    _require_equal(full.get("expected_cells"), 60, "experiment_a.full_grid.expected_cells")
    _require_equal(full.get("expected_pair_units"), 300, "experiment_a.full_grid.expected_pair_units")
    _require_equal(fallback.get("expected_cells"), 36, "experiment_a.fallback_grid.expected_cells")
    _require_equal(fallback.get("expected_pair_units"), 180, "experiment_a.fallback_grid.expected_pair_units")

    experiment_b = _mapping(raw, "experiment_b_epsilon_sweep", "root")
    _require_locked_subset(
        experiment_b, _LOCKED_EXPERIMENT_B, "experiment_b_epsilon_sweep"
    )
    _require_equal(tuple(experiment_b.get("methods", [])), ("fgsm", "pgd"), "experiment_b.methods")
    _require_equal(tuple(experiment_b.get("scoring_conditions", [])), _SCORING_CONDITIONS, "experiment_b.scoring_conditions")
    epsilon_values = _sequence(experiment_b, "epsilons", "experiment_b")
    if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in epsilon_values):
        raise ConfigError("epsilon grid values must be finite numbers")
    epsilons = tuple(float(value) for value in epsilon_values)
    if len(epsilons) != len(set(epsilons)):
        raise ConfigError("epsilon grid contains duplicate values")
    if list(epsilons) != sorted(epsilons):
        raise ConfigError("epsilon grid must be strictly increasing")
    _require_equal(epsilons, _LOCKED_EPSILONS, "experiment_b.epsilon grid")
    required_scope = _mapping(experiment_b, "required_scope", "experiment_b")
    _require_equal(required_scope.get("expected_cells"), 12, "experiment_b.required_scope.expected_cells")

    experiment_c = _mapping(raw, "experiment_c_orientation_redecodability", "root")
    _require_locked_subset(
        experiment_c,
        _LOCKED_CONSTRUCT,
        "experiment_c_orientation_redecodability",
    )
    cell = _mapping(experiment_c, "cell", "experiment_c")
    _require_equal((cell.get("model_key"), cell.get("task"), cell.get("layer")), ("qwen", "sst2", 14), "experiment_c.cell")
    candidate = _mapping(experiment_c, "candidate_scope", "experiment_c")
    _require_equal(tuple(candidate.get("architectures", [])), ("linear", "mlp", "mka"), "experiment_c.candidate_scope.architectures")
    _require_equal(tuple(candidate.get("seeds", [])), tuple(range(20)), "experiment_c.candidate_scope.seeds")
    _require_equal(candidate.get("expected_candidates"), 60, "experiment_c.candidate_scope.expected_candidates")

    outputs = _mapping(raw, "outputs", "root")
    _require_locked_subset(outputs, _LOCKED_OUTPUTS, "outputs")
    required_files = _sequence(outputs, "required_files", "outputs")
    for required in ("run_manifest.json", "console.log", "analysis_summary.json", "manuscript_numbers.tex"):
        if required not in required_files:
            raise ConfigError(f"outputs.required_files is missing {required}")
    figures = _mapping(outputs, "figures", "outputs")
    _require_equal(tuple(figures.get("formats", [])), ("pdf", "png"), "outputs.figures.formats")
    _require_equal(figures.get("png_dpi"), 300, "outputs.figures.png_dpi")

    paper_update = _mapping(raw, "paper_update", "root")
    _require_locked_subset(paper_update, _LOCKED_PAPER_UPDATE, "paper_update")


def load_revision_config(path: str | Path) -> RevisionConfig:
    """Load the supplied YAML, reject duplicate keys, and validate locked choices."""

    source_path = Path(path)
    try:
        raw = yaml.load(source_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except ConfigError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"could not load revision config: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("revision config root must be a mapping")
    _validate_config(raw)
    serialized = canonical_json(raw)
    config = RevisionConfig(
        raw=raw,
        source_path=source_path.resolve(),
        canonical_json=serialized,
        config_hash=sha256_json(raw),
    )

    expansions = {
        "full cells": config.matched_split_cells("full"),
        "fallback cells": config.matched_split_cells("fallback"),
        "full rows": config.matched_split_row_keys("full"),
        "fallback rows": config.matched_split_row_keys("fallback"),
        "epsilon rows": config.epsilon_sweep_row_keys(),
        "construct edits": config.construct_edit_keys(),
    }
    expected_counts = {
        "full cells": 60,
        "fallback cells": 36,
        "full rows": 2_400,
        "fallback rows": 1_440,
        "epsilon rows": 2_400,
        "construct edits": 65,
    }
    for label, keys in expansions.items():
        if len(keys) != len(set(keys)):
            raise ConfigError(f"duplicate generated {label} keys")
        if len(keys) != expected_counts[label]:
            raise ConfigError(
                f"generated {label} count {len(keys)} does not match {expected_counts[label]}"
            )
    return config


__all__ = [
    "ConfigError",
    "ConstructEditKey",
    "EpsilonSweepRowKey",
    "MatchedSplitCellKey",
    "MatchedSplitRowKey",
    "RevisionConfig",
    "load_revision_config",
]
