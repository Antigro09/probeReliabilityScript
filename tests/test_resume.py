"""
Tests for WS0.1: resume-safety of the benchmark runner.

v1 opened its output in append mode with no completed-cell check, so an
interrupted-and-restarted run silently duplicated rows and poisoned every
median. v2 reads the existing file on startup and skips completed
(layer, arch, seed) cells; it also refuses mixed-schema and
already-duplicated files instead of building on them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from scripts.run_benchmark import load_completed_cells, SCHEMA_VERSION


def _row(layer, arch, seed, schema=SCHEMA_VERSION):
    return json.dumps({"layer": layer, "arch": arch, "seed": seed,
                       "schema_version": schema})


def test_missing_file_means_nothing_done(tmp_path):
    assert load_completed_cells(tmp_path / "nope.jsonl") == set()


def test_completed_cells_are_read(tmp_path):
    p = tmp_path / "out.jsonl"
    p.write_text("\n".join([
        _row(1, "linear", 1000),
        _row(1, "linear", 1001),
        _row(3, "mka", 1000),
    ]) + "\n")
    done = load_completed_cells(p)
    assert done == {(1, "linear", 1000), (1, "linear", 1001), (3, "mka", 1000)}


def test_v1_rows_are_refused(tmp_path):
    p = tmp_path / "out.jsonl"
    # v1 rows carry no schema_version at all.
    p.write_text(json.dumps({"layer": 1, "arch": "linear", "seed": 1000}) + "\n")
    with pytest.raises(SystemExit, match="schema"):
        load_completed_cells(p)


def test_duplicated_cells_are_refused(tmp_path):
    p = tmp_path / "out.jsonl"
    p.write_text("\n".join([_row(1, "mlp", 1000), _row(1, "mlp", 1000)]) + "\n")
    with pytest.raises(SystemExit, match="duplicate"):
        load_completed_cells(p)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
