from __future__ import annotations

import pytest
import torch

from src.probes import LinearProbe
from src.reviewer_revision.experiments import (
    assert_disjoint_named_groups,
    candidate_rank_one_direction,
    evaluate_fixed_evaluator_edit,
    rank_one_projection_edit,
)
from src.reviewer_revision.runner import (
    _fresh_checkpoint,
    _validate_rank_one_reconstruction,
)
from src.reviewer_revision.training import DecoderSpec


def _perfect_linear_probe() -> LinearProbe:
    probe = LinearProbe(2)
    with torch.no_grad():
        probe.linear.weight.copy_(torch.tensor([[-1.0, 0.0], [1.0, 0.0]]))
        probe.linear.bias.zero_()
    return probe


def test_rank_one_projection_removes_selected_direction():
    X = torch.tensor([[2.0, 3.0], [-4.0, 5.0], [1.0, -2.0]])
    direction = torch.tensor([1.0, 0.0])
    edited, diagnostics = rank_one_projection_edit(X, direction)
    assert torch.allclose(edited[:, 0], torch.zeros(3), atol=1e-7)
    assert torch.allclose(edited[:, 1], X[:, 1])
    assert diagnostics["maximum_absolute_residual"] <= 1e-7
    assert diagnostics["rms_residual"] <= 1e-7


def test_rank_one_projection_residual_threshold_is_a_hard_gate():
    generator = torch.Generator().manual_seed(4)
    X = torch.randn(31, 7, generator=generator)
    direction = torch.randn(7, generator=generator)

    with pytest.raises(ValueError, match="residual threshold"):
        rank_one_projection_edit(X, direction, residual_tolerance=0.0)


def test_rank_one_projection_uses_stable_arithmetic_before_float32_storage():
    """Float32 cancellation must not reject a valid high-dimensional edit."""

    generator = torch.Generator().manual_seed(0)
    X = torch.randn(304, 1536, generator=generator) * 10.0
    direction = torch.randn(1536, generator=generator)

    edited, diagnostics = rank_one_projection_edit(X, direction)

    assert edited.dtype == torch.float32
    assert diagnostics["maximum_absolute_residual"] <= 1.0e-5
    assert diagnostics["residual_gate_passed"] is True


def test_linear_jacobian_direction_agrees_with_weight_difference():
    probe = _perfect_linear_probe()
    X = torch.tensor([[-2.0, 0.5], [-1.0, -0.2], [1.0, 0.3], [2.0, -0.4]])
    direction, diagnostics = candidate_rank_one_direction(
        probe, X, device=torch.device("cpu")
    )
    assert direction.shape == (2,)
    assert diagnostics["linear_weight_cosine"] > 0.999


def test_all_named_construct_groups_are_disjoint():
    groups = {
        "candidate": {"c1", "c2"},
        "evaluator": {"e1", "e2"},
        "direction_fit": {"d1"},
        "fresh_decoder_fit": {"f1", "f2"},
        "orientation_calibration": {"o1"},
        "final_test": {"t1"},
        "unused_phase2_test": {"u1"},
    }
    assert_disjoint_named_groups(groups)
    groups["final_test"].add("d1")
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_named_groups(groups)


def test_fixed_evaluator_orientation_is_fit_on_calibration_and_frozen():
    probe = _perfect_linear_probe()
    X_calibration_pre = torch.tensor([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    X_final_pre = X_calibration_pre.clone()
    labels = torch.tensor([0, 0, 1, 1])
    # The edit inverts only the target coordinate. Calibration should select -1,
    # then the exact sign is reused on final without allowing a second fit.
    X_calibration_post = -X_calibration_pre
    X_final_post = -X_final_pre
    result = evaluate_fixed_evaluator_edit(
        probe=probe,
        X_calibration_pre=X_calibration_pre,
        X_calibration_post=X_calibration_post,
        calibration_labels=labels,
        X_final_pre=X_final_pre,
        X_final_post=X_final_post,
        final_labels=labels,
        device=torch.device("cpu"),
    )
    assert result["orientation"]["sign"] == -1
    assert result["final_post_raw"]["accuracy"] == 0.0
    assert result["final_post_oriented"]["accuracy"] == 1.0
    assert result["C_raw"] == 1.0
    assert result["C_orientation"] == 0.0


def test_fresh_decoder_checkpoint_saves_final_per_example_outputs(tmp_path):
    generator = torch.Generator().manual_seed(44)
    X = torch.randn(36, 4, generator=generator)
    y = (X[:, 0] > 0).long()
    checkpoint = tmp_path / "fresh.pt"
    _, checkpoint_hash, _, _, per_example_path = _fresh_checkpoint(
        checkpoint,
        X_train=X[:20],
        y_train=y[:20],
        X_validation=X[20:28],
        y_validation=y[20:28],
        X_final=X[28:],
        y_final=y[28:],
        final_example_ids=[f"final-{index}" for index in range(8)],
        spec=DecoderSpec(
            architecture="linear",
            learning_rate=0.01,
            weight_decay=0.0,
            epochs=3,
            batch_size=8,
            patience=2,
        ),
        seed=0,
        device=torch.device("cpu"),
        metadata={"edit_id": "tiny"},
    )

    payload = torch.load(per_example_path, weights_only=True)
    assert payload["checkpoint_sha256"] == checkpoint_hash
    assert payload["example_ids"] == [f"final-{index}" for index in range(8)]
    assert payload["split_names"] == ["final_test"] * 8
    assert payload["logits"].shape == (8, 2)
    assert payload["probabilities"].shape == (8, 2)
    assert payload["predictions"].shape == (8,)
    assert torch.equal(payload["labels"], y[28:])


def test_rank_one_recipe_validation_reconstructs_representations_and_logits():
    X = {
        "decoder": torch.tensor([[1.0, 2.0], [-1.0, 0.5]]),
        "calibration": torch.tensor([[2.0, -1.0]]),
        "final": torch.tensor([[0.5, 3.0], [-0.5, -2.0]]),
    }
    direction = torch.tensor([1.0, 0.0])
    edited = {name: rank_one_projection_edit(value, direction)[0] for name, value in X.items()}
    diagnostics = _validate_rank_one_reconstruction(
        source_subsets=X,
        saved_edited_subsets=edited,
        direction=direction,
        validation_probe=_perfect_linear_probe(),
        device=torch.device("cpu"),
        tolerance=1.0e-6,
    )
    assert diagnostics["maximum_absolute_representation_difference"] == 0.0
    assert diagnostics["maximum_absolute_logit_difference"] == 0.0
    assert len(diagnostics["reconstructed_final_logits_sha256"]) == 64

    broken = {**edited, "final": edited["final"] + 0.01}
    with pytest.raises(ValueError, match="reconstruction"):
        _validate_rank_one_reconstruction(
            source_subsets=X,
            saved_edited_subsets=broken,
            direction=direction,
            validation_probe=_perfect_linear_probe(),
            device=torch.device("cpu"),
            tolerance=1.0e-6,
        )
