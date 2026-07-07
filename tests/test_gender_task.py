"""
Tests for WS2: the hardened gender task loader.

Every failure signature of the v1 dataset must now be a hard error:
missing file (no silent synthetic fallback), NEUTRAL occupation labels,
sentences carrying both gender labels, and sentences that do not contain
their pronoun.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.tasks import get_task

VALID_V2_LINES = [
    "The engineer told someone that she had solved the problem.\tFEM\tMASC_SKEW",
    "The engineer told someone that he had solved the problem.\tMASC\tMASC_SKEW",
    "The nurse told the patient that she would return soon.\tFEM\tFEM_SKEW",
    "The nurse told the patient that he would return soon.\tMASC\tFEM_SKEW",
]


def _write(tmp_path: Path, lines: list[str]) -> Path:
    p = tmp_path / "gender_test.tsv"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_missing_file_is_hard_error_no_synthetic_fallback(tmp_path):
    task = get_task("gender")
    with pytest.raises(FileNotFoundError, match="no synthetic fallback"):
        task.load([tmp_path / "does_not_exist.tsv"], max_examples=None, seed=0)
    # The v1 silent generator must be gone entirely.
    assert not hasattr(task, "_generate_synthetic")


def test_valid_v2_data_loads_with_correct_mapping(tmp_path):
    task = get_task("gender")
    examples = task.load([_write(tmp_path, VALID_V2_LINES)],
                         max_examples=None, seed=0)
    assert len(examples) == 4
    by_sentence = {ex.sentence: ex for ex in examples}
    fem = by_sentence["The nurse told the patient that she would return soon."]
    assert (fem.zc, fem.ze) == (1, 1)
    masc = by_sentence["The engineer told someone that he had solved the problem."]
    assert (masc.zc, masc.ze) == (0, 0)


def test_neutral_label_rejected(tmp_path):
    lines = VALID_V2_LINES + [
        "The doctor said that she was ready.\tFEM\tNEUTRAL",
    ]
    task = get_task("gender")
    with pytest.raises(ValueError, match="NEUTRAL"):
        task.load([_write(tmp_path, lines)], max_examples=None, seed=0)


def test_both_labels_per_sentence_rejected(tmp_path):
    # Same sentence string with both labels (contains both pronouns so the
    # per-row pronoun check cannot mask the duplicate-label check).
    lines = [
        "The plumber told her and him the truth.\tFEM\tMASC_SKEW",
        "The plumber told her and him the truth.\tMASC\tMASC_SKEW",
    ]
    task = get_task("gender")
    with pytest.raises(ValueError, match="BOTH gender labels"):
        task.load([_write(tmp_path, lines)], max_examples=None, seed=0)


def test_v1_truncated_prefix_rejected(tmp_path):
    # v1 style: prefix ends before the pronoun -> pronoun missing from input.
    lines = ["The technician told the customer that\tFEM\tMASC_SKEW"]
    task = get_task("gender")
    with pytest.raises(ValueError, match="pronoun"):
        task.load([_write(tmp_path, lines)], max_examples=None, seed=0)


def test_repo_gender_data_is_valid_v2():
    """The gender.tsv shipped in data/ has been REGENERATED to the valid v2
    artifact (full-sentence, pronoun included, no NEUTRAL, no both-labels rows),
    so the strict v2 loader must accept it and its validation must pass. The
    rejection guards for broken data are covered by the tmp_path tests above."""
    f = PROJECT_ROOT / "data" / "gender.tsv"
    if not f.exists():
        pytest.skip("gender data file not present")
    task = get_task("gender")
    examples = task.load([f], max_examples=None, seed=0)
    assert len(examples) > 0
    # every example carries a pronoun consistent with its label and one zc per sentence
    zc_by_sentence = {}
    for ex in examples:
        assert ex.zc in (0, 1) and ex.ze in (0, 1)
        zc_by_sentence.setdefault(ex.sentence, set()).add(ex.zc)
    assert all(len(v) == 1 for v in zc_by_sentence.values())


def test_malformed_line_rejected(tmp_path):
    lines = VALID_V2_LINES + ["only two\tfields"]
    task = get_task("gender")
    with pytest.raises(ValueError, match="3 tab-separated"):
        task.load([_write(tmp_path, lines)], max_examples=None, seed=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
