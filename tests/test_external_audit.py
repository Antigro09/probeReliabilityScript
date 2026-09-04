"""Tests for the generic external scored-object audit."""

from scripts.audit_external_results import audit


def test_audit_detects_candidate_independent_target_and_max_gap():
    rows = []
    for layer in (1, 2):
        for arch, seed in (("linear", 1), ("mlp", 2), ("mka", 3)):
            rows.append({
                "model": "m", "layer": layer, "task": "t",
                "arch": arch, "seed": seed, "R": 0.9,
                "methods": {"a": 0.9, "b": 0.1},
            })
    result = audit(
        rows, ["model", "layer", "task"], ["arch", "seed"], "R",
        per_method_field="methods",
    )
    assert result["fraction_cells_candidate_independent"] == 1.0
    assert result["distinct_target_histogram"] == {"1": 2}
    assert result["per_method"]["max_minus_mean_median"] == 0.4
