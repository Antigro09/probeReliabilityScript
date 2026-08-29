from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.reviewer_revision.config import (
    ConfigError,
    ConstructEditKey,
    EpsilonSweepRowKey,
    MatchedSplitCellKey,
    load_revision_config,
)

SPEC_PATH = Path(__file__).resolve().parents[2] / "revision_experiment_spec.yaml"


def _write_mutated_spec(
    tmp_path: Path, path: tuple[str | int, ...], replacement: Any
) -> Path:
    payload = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    cursor: Any = payload
    for component in path[:-1]:
        cursor = cursor[component]
    cursor[path[-1]] = copy.deepcopy(replacement)
    destination = tmp_path / "mutated.yaml"
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return destination


def _path_label(path: tuple[str | int, ...]) -> str:
    label = ""
    for component in path:
        if isinstance(component, int):
            label += f"[{component}]"
        else:
            label += ("." if label else "") + component
    return label


def test_locked_config_expands_exact_experiment_keys() -> None:
    config = load_revision_config(SPEC_PATH)

    full_cells = config.matched_split_cells("full")
    fallback_cells = config.matched_split_cells("fallback")
    epsilon_rows = config.epsilon_sweep_row_keys()
    construct_edits = config.construct_edit_keys()

    assert len(full_cells) == 60
    assert len(set(full_cells)) == 60
    assert MatchedSplitCellKey("qwen", "sst2", 14) in full_cells

    assert len(fallback_cells) == 36
    assert len(set(fallback_cells)) == 36
    assert {
        cell.layer
        for cell in fallback_cells
        if cell.model_key == "gemma" and cell.task == "sva"
    } == {1, 14, 26}

    assert len(config.matched_split_row_keys("full")) == 2_400
    assert len(config.matched_split_row_keys("fallback")) == 1_440

    assert len(epsilon_rows) == 2_400
    assert len(set(epsilon_rows)) == 2_400
    assert EpsilonSweepRowKey("qwen", "sst2", 14, 0, "pgd", 0.5, "split") in epsilon_rows

    assert len(construct_edits) == 65
    assert len(set(construct_edits)) == 65
    assert len([key for key in construct_edits if key.edit_kind == "alterrep"]) == 5
    assert len([key for key in construct_edits if key.edit_kind == "dcand_crossfit"]) == 60
    assert ConstructEditKey("dcand_crossfit", "mka", 19) in construct_edits


def test_locked_config_has_stable_canonical_hash() -> None:
    first = load_revision_config(SPEC_PATH)
    second = load_revision_config(SPEC_PATH)

    assert first.config_hash == second.config_hash
    assert len(first.config_hash) == 64
    assert first.canonical_json == second.canonical_json
    assert first.raw["reproducibility"]["pair_seeds"] == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    "epsilons",
    [
        [0.0, 0.1, 0.1],
        [0.0, 0.5, 0.25],
        [0.001953125, 0.00390625, 0.5],
        [0.0, float("nan"), 0.5],
    ],
)
def test_invalid_epsilon_grids_are_rejected(tmp_path: Path, epsilons: list[float]) -> None:
    payload = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    payload["experiment_b_epsilon_sweep"]["epsilons"] = epsilons
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError, match="epsilon"):
        load_revision_config(path)


def test_duplicate_yaml_mapping_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text(
        SPEC_PATH.read_text(encoding="utf-8") + "\nschema_version: 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="duplicate.*schema_version"):
        load_revision_config(path)


def test_duplicate_generated_keys_are_rejected(tmp_path: Path) -> None:
    payload = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    payload["models"][0]["layers"] = [1, 4, 6, 6, 12]
    path = tmp_path / "duplicate-layer.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError, match="duplicate.*layer"):
        load_revision_config(path)


def test_declared_expected_counts_are_validated(tmp_path: Path) -> None:
    payload = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    payload["experiment_a_expanded_matched_split"]["full_grid"]["expected_cells"] = 59
    path = tmp_path / "wrong-count.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError, match="expected_cells"):
        load_revision_config(path)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("reproducibility", "master_seed"), 17),
        (("reproducibility", "bootstrap_seed"), 18),
        (("reproducibility", "permutation_seed"), 19),
        (("reproducibility", "float_dtype_for_probe_training"), "float64"),
        (("reproducibility", "deterministic_algorithms"), False),
        (("reproducibility", "preserve_existing_artifacts"), False),
        (("runtime", "transformer_extraction_device"), "cpu"),
        (("runtime", "linear_probe_device"), "mps"),
        (("runtime", "statistics_device"), "mps"),
        (("runtime", "epsilon_all_layers_max_projected_hours"), 13),
        (("runtime", "env", "PYTORCH_ENABLE_MPS_FALLBACK"), "0"),
        (("runtime", "env", "TOKENIZERS_PARALLELISM"), "true"),
        (("models", 0, "hf_id"), "EleutherAI/pythia-410m"),
        (("models", 0, "layers"), [2, 4, 6, 9, 12]),
        (("models", 0, "middle_layer"), 4),
        (("models", 1, "hf_id"), "openai-community/gpt2-medium"),
        (("models", 2, "hf_id"), "google-bert/bert-large-uncased"),
        (("models", 3, "hf_id"), "Qwen/Qwen2.5-3B"),
        (("models", 4, "hf_id"), "google/gemma-2-9b"),
        (("models", 5, "hf_id"), "meta-llama/Llama-3.2-1B"),
        (("tasks", "excluded"), []),
        (("tasks", "exclusion_reason"), "different rationale"),
        (("shared_score_definition", "target_damage"), "different formula"),
        (("shared_score_definition", "control_preservation"), "different formula"),
        (("shared_score_definition", "harmonic_mean"), "different formula"),
        (("baseline_interventions", "fgsm", "norm"), "l2"),
        (("baseline_interventions", "pgd", "norm"), "l2"),
        (("baseline_interventions", "pgd", "step_size"), 0.2),
        (("baseline_interventions", "pgd", "step_size_rule_for_sweep"), "epsilon / 4"),
        (("experiment_a_expanded_matched_split", "full_grid", "tasks"), ["sst2"]),
        (("experiment_a_expanded_matched_split", "full_grid", "layer_policy"), "middle_only"),
        (("experiment_a_expanded_matched_split", "full_grid", "expected_model_task_blocks"), 11),
        (("experiment_a_expanded_matched_split", "fallback_grid", "trigger"), "manual"),
        (("experiment_a_expanded_matched_split", "hard_floor", "minimum_total_cells"), 35),
        (("experiment_a_expanded_matched_split", "estimand"), "different estimand"),
        (("experiment_a_expanded_matched_split", "inference", "primary_cluster"), "cell"),
        (("experiment_a_expanded_matched_split", "inference", "bootstrap_hierarchy"), ["pair"]),
        (("experiment_b_epsilon_sweep", "required_scope", "layers"), "all"),
        (("experiment_b_epsilon_sweep", "automatic_extension", "condition"), "always"),
        (("experiment_b_epsilon_sweep", "pgd", "steps"), 9),
        (("experiment_b_epsilon_sweep", "pgd", "step_size_rule"), "epsilon / 4"),
        (("experiment_b_epsilon_sweep", "paired_randomness", "reuse_pgd_random_start_base_noise_across_epsilons"), False),
        (("experiment_b_epsilon_sweep", "integrity", "epsilon_zero_max_abs_representation_delta_tolerance"), 1.0e-5),
        (("experiment_b_epsilon_sweep", "integrity", "baseline_epsilon_must_reproduce_archived_pattern"), False),
        (("experiment_c_orientation_redecodability", "cell", "model_id"), "Qwen/Qwen2.5-3B"),
        (("experiment_c_orientation_redecodability", "cell", "selection_rule"), "post-hoc selection"),
        (("experiment_c_orientation_redecodability", "edit_objects", "required"), ["alterrep"]),
        (("experiment_c_orientation_redecodability", "source_split", "preserve_phase2_candidate_fraction"), 0.3),
        (("experiment_c_orientation_redecodability", "source_split", "subdivide_phase2_intervention_fraction", "final_test"), "one_fifth"),
        (("experiment_c_orientation_redecodability", "source_split", "grouping_unit"), "example_id"),
        (("experiment_c_orientation_redecodability", "direction_estimation", "normalize_direction_l2"), False),
        (("experiment_c_orientation_redecodability", "fixed_evaluator_metrics", "orientation", "choose_identity_or_flip_on"), "final_test"),
        (("experiment_c_orientation_redecodability", "fresh_decoders", "train_on"), "candidate"),
        (("experiment_c_orientation_redecodability", "fresh_decoders", "hyperparameter_selection"), "retune per edit"),
        (("experiment_c_orientation_redecodability", "fresh_decoders", "required_models", "linear", "learning_rates"), [0.001]),
        (("experiment_c_orientation_redecodability", "fresh_decoders", "required_models", "linear", "epochs"), 49),
        (("experiment_c_orientation_redecodability", "fresh_decoders", "required_models", "mlp", "hidden_dim"), 128),
        (("experiment_c_orientation_redecodability", "fresh_decoders", "required_models", "mlp", "seeds"), [0, 1]),
        (("experiment_c_orientation_redecodability", "fresh_decoders", "derived_quantities", "target_recovery_ratio"), "different formula"),
        (("experiment_c_orientation_redecodability", "save"), ["direction_vectors"]),
        (("experiment_c_orientation_redecodability", "interpretation_rules", "inversion"), "different rule"),
        (("outputs", "root"), "results/other"),
        (("outputs", "immutable_run_directory"), False),
        (("outputs", "required_files"), ["run_manifest.json", "console.log"]),
        (("outputs", "figures", "directory"), "plots"),
        (("outputs", "figures", "names", "epsilon_sweep"), "different_name"),
        (("paper_update", "source"), "other.tex"),
        (("paper_update", "required_markers"), ["one marker"]),
        (("paper_update", "terminology", "never_equate_fixed_decoder_damage_with_erasure"), False),
        (("paper_update", "compile", "fail_on_missing_citations"), False),
    ],
)
def test_all_locked_scientific_runtime_and_output_values_are_validated(
    tmp_path: Path, path: tuple[str | int, ...], replacement: Any
) -> None:
    mutated = _write_mutated_spec(tmp_path, path, replacement)

    with pytest.raises(ConfigError, match=re.escape(_path_label(path))):
        load_revision_config(mutated)


def test_semantically_identical_yaml_with_reordered_mapping_keys_is_accepted(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    path = tmp_path / "reordered.yaml"
    path.write_text(
        "# Formatting and key order are deliberately different.\n"
        + yaml.safe_dump(payload, sort_keys=True),
        encoding="utf-8",
    )

    reordered = load_revision_config(path)
    original = load_revision_config(SPEC_PATH)

    assert reordered.raw == original.raw
    assert reordered.config_hash == original.config_hash


def test_unknown_locked_parameter_is_rejected_instead_of_silently_ignored(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    payload["runtime"]["unrecognized_fallback_limit"] = 1
    path = tmp_path / "unknown-parameter.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError, match="runtime.unrecognized_fallback_limit"):
        load_revision_config(path)


def test_boolean_locks_do_not_accept_equal_integer_values(tmp_path: Path) -> None:
    path = _write_mutated_spec(
        tmp_path, ("reproducibility", "deterministic_algorithms"), 1
    )

    with pytest.raises(ConfigError, match="deterministic_algorithms"):
        load_revision_config(path)
