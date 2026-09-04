"""Schema-v3 analysis extraction tests."""

import pytest

pytest.importorskip("pandas")

from scripts.ws9_analyze import candidate_table, layer_tables


def _scores(auc):
    return {
        "n": 20, "n_positive": 10,
        "accuracy": auc, "complemented_accuracy": auc,
        "balanced_accuracy": auc, "complemented_balanced_accuracy": auc,
        "roc_auc": auc, "oriented_roc_auc": auc,
        "log_loss": 0.5, "oriented_log_loss": 0.5,
        "brier": 0.2, "oriented_brier": 0.2,
    }


def test_candidate_table_recomputes_auc_floor_sensitivity():
    evaluator = {
        "target_pre": _scores(0.90), "target_post": _scores(0.50),
        "control_pre": _scores(0.90), "control_post": _scores(0.85),
    }
    row = {
        "model": "m", "task": "sva", "position": "last",
        "layer": 1, "arch": "linear", "seed": 1000,
        "deep_evaluated": True, "candidate_accuracy": 0.8,
        "num_params": 10,
        "hessian": {
            "A_raw_mean_abs_cosine": 0.1,
            "A_subspace": {"1": 0.2, "5": 0.4},
            "random_null": {"1": {"z": 2.0}, "5": {"z": 1.0}},
            "eigenpair_residuals": [{"relative_residual": 0.01}],
        },
        "direction_stability": {"median": 0.8},
        "candidate_evaluator_agreement": {"median": 0.7},
        "crossfit": {"fixed": {"aggregate": {"R_auc_oriented": 0.6}}},
        "edits": {
            "candidate_r1": {
                "evaluation": {
                    "fixed": {
                        "aggregate": {
                            "R_auc_oriented": 0.9,
                            "C_auc_oriented": 1.0,
                            "S_auc_oriented": 0.85,
                        },
                        "per_evaluator": [evaluator],
                    },
                    "retrained_summary": {
                        "target": {"max_oriented_roc_auc": 0.55},
                        "control": {"max_oriented_roc_auc": 0.85},
                    },
                }
            }
        },
    }
    table, ks = candidate_table([row], [0.55, 0.60])
    assert ks == [1, 5]
    assert table.loc[0, "robust_erasure"] == pytest.approx(0.9)
    assert table.loc[0, "candidate_evaluator_agreement"] == pytest.approx(0.7)
    assert table.loc[0, "R_fixed_auc_floor_0p55"] is not None
    assert table.loc[0, "R_fixed_auc_floor_0p60"] is not None


def test_layer_tables_keep_exclusion_reasons():
    inclusion, baselines, sweep = layer_tables([
        {
            "model": "m", "task": "gender", "position": "pre-pronoun",
            "layer": 2, "status": "excluded", "reason": "learnability_gate_failed",
        }
    ])
    assert inclusion.loc[0, "reason"] == "learnability_gate_failed"
    assert baselines.empty
    assert sweep.empty
