"""
Task abstraction for cross-task probe reliability evaluation.

A Task encapsulates everything that distinguishes one probing problem from
another: dataset loading, target label Zc, selectivity feature Ze, chance
accuracy, and metadata. The pipeline accepts any Task and otherwise behaves
identically.

Three tasks are implemented in this module:
    - SVATask:    subject-verb agreement (Linzen et al. 2016)
    - GenderTask: gender agreement (Winogender-style templates)
    - SST2Task:   binary sentiment with sentence-length selectivity feature

All tasks produce a list[Example] where each Example has:
    sentence (str), zc (int), ze (int)
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Import the existing LinzenExample dataclass for SVA backward compatibility
from .data import (
    LinzenExample, build_examples as build_sva_examples,
    split_balanced as split_balanced_sva,
)


# ---------------------------------------------------------------------------
# Generic Example
# ---------------------------------------------------------------------------

@dataclass
class Example:
    """Generic probing example. Backwards-compatible with LinzenExample."""
    sentence: str
    zc: int
    ze: int

    def to_dict(self) -> dict:
        return {"sentence": self.sentence, "zc": self.zc, "ze": self.ze}


# ---------------------------------------------------------------------------
# Task base class
# ---------------------------------------------------------------------------

class Task(ABC):
    """Base class for a probing task."""

    name: str = "abstract"
    chance_accuracy: float = 0.5
    zc_description: str = ""
    ze_description: str = ""

    @abstractmethod
    def load(self, paths: Iterable[Path], max_examples: int | None,
             seed: int) -> list[Example]:
        """Return a list of Examples with .sentence, .zc, .ze populated."""

    def split(self, examples: list[Example],
              val_frac: float = 0.4, inter_frac: float = 0.4,
              seed: int = 42) -> tuple[list[Example], list[Example], list[Example]]:
        """Default: balanced split across (zc, ze) cells.
        Subclasses can override if their feature combinatorics differ."""
        return _split_balanced_4cell(examples, val_frac, inter_frac, seed)


def _split_balanced_4cell(examples, val_frac, inter_frac, seed):
    """Generic version of split_balanced for any (zc, ze) binary feature pair."""
    if val_frac + inter_frac >= 1.0:
        raise ValueError("val_frac + inter_frac must be < 1.0")
    rng = random.Random(seed)
    buckets: dict[tuple[int, int], list[Example]] = {
        (0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []
    }
    for ex in examples:
        buckets[(ex.zc, ex.ze)].append(ex)
    min_n = min(len(v) for v in buckets.values())
    if min_n == 0:
        # Some cell is empty; fall back to (zc-only) balancing.
        return _split_balanced_zc_only(examples, val_frac, inter_frac, seed)
    balanced: list[Example] = []
    for v in buckets.values():
        rng.shuffle(v)
        balanced.extend(v[:min_n])
    rng.shuffle(balanced)
    n = len(balanced)
    n_val = int(val_frac * n)
    n_int = int(inter_frac * n)
    return (balanced[:n_val],
            balanced[n_val:n_val + n_int],
            balanced[n_val + n_int:])


def _split_balanced_zc_only(examples, val_frac, inter_frac, seed):
    """Fallback when (zc, ze) cells are uneven enough that 4-cell balance fails."""
    rng = random.Random(seed)
    by_zc: dict[int, list[Example]] = {}
    for ex in examples:
        by_zc.setdefault(ex.zc, []).append(ex)
    min_n = min(len(v) for v in by_zc.values())
    balanced: list[Example] = []
    for v in by_zc.values():
        rng.shuffle(v)
        balanced.extend(v[:min_n])
    rng.shuffle(balanced)
    n = len(balanced)
    n_val = int(val_frac * n)
    n_int = int(inter_frac * n)
    return (balanced[:n_val],
            balanced[n_val:n_val + n_int],
            balanced[n_val + n_int:])


# ---------------------------------------------------------------------------
# SVA task (existing)
# ---------------------------------------------------------------------------

class SVATask(Task):
    """Subject-verb agreement (Linzen et al. 2016).
    Zc = grammatical number of target verb (1 sg / 0 pl).
    Ze = grammatical number of last noun in prefix.
    """

    name = "sva"
    chance_accuracy = 0.5
    zc_description = "grammatical number of the target verb (singular / plural)"
    ze_description = "grammatical number of the last noun in the prefix"

    def load(self, paths, max_examples, seed):
        # Re-use the existing build_examples; LinzenExample is duck-compatible
        # with Example since it has .sentence, .zc, .ze and to_dict().
        linzen_examples = build_sva_examples(paths, max_examples=max_examples,
                                             seed=seed)
        return [Example(sentence=e.sentence, zc=e.zc, ze=e.ze)
                for e in linzen_examples]


# ---------------------------------------------------------------------------
# Gender agreement task
# ---------------------------------------------------------------------------

class GenderTask(Task):
    """Gender agreement using Winogender-style templated sentences.

    Each example has a profession noun (gendered or neutral by US Census
    statistics), a pronoun referring back, and we predict whether the
    pronoun is feminine or masculine. Selectivity feature Ze is the
    profession's gender association (skews toward female / male in
    training data).

    Data format per file line: TAB-separated
        sentence_prefix\tpronoun_label\tprofession_label
    where pronoun_label in {"FEM", "MASC"} and profession_label in
    {"FEM_SKEW", "MASC_SKEW", "NEUTRAL"}.

    If the dataset isn't present, this task uses a built-in template
    generator that produces ~2000 balanced examples deterministically.
    """

    name = "gender"
    chance_accuracy = 0.5
    zc_description = "gender of the pronoun (feminine / masculine)"
    ze_description = "occupation's stereotypical gender (female-skew / male-skew)"

    def load(self, paths, max_examples, seed):
        paths = [Path(p) for p in paths]
        # Try to load from disk first
        if any(p.exists() for p in paths):
            examples = self._load_from_files(paths)
        else:
            examples = self._generate_synthetic(seed)
        rng = random.Random(seed)
        rng.shuffle(examples)
        if max_examples is not None:
            examples = examples[:max_examples]
        return examples

    def _load_from_files(self, paths):
        examples: list[Example] = []
        for path in paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) != 3:
                        continue
                    sent, pron_lab, prof_lab = parts
                    if pron_lab not in {"FEM", "MASC"}:
                        continue
                    if prof_lab not in {"FEM_SKEW", "MASC_SKEW", "NEUTRAL"}:
                        continue
                    zc = 1 if pron_lab == "FEM" else 0
                    # Ze: 1 = female-skewing or neutral, 0 = male-skewing
                    ze = 1 if prof_lab != "MASC_SKEW" else 0
                    examples.append(Example(sentence=sent, zc=zc, ze=ze))
        return examples

    def _generate_synthetic(self, seed: int) -> list[Example]:
        """Generate templated gender-agreement examples deterministically.
        This guarantees the experiment can run without external data and the
        result is reproducible."""
        rng = random.Random(seed)

        # US Census-style profession lists (illustrative only; for the paper
        # we'll cite the source if we use the actual Winogender data).
        fem_skew = ["nurse", "secretary", "teacher", "librarian", "stylist",
                    "receptionist", "nanny", "hairdresser", "dietitian",
                    "paralegal"]
        masc_skew = ["plumber", "mechanic", "carpenter", "electrician",
                     "firefighter", "construction worker", "truck driver",
                     "engineer", "pilot", "soldier"]
        neutral = ["doctor", "scientist", "writer", "artist", "lawyer",
                   "manager", "consultant", "student", "researcher", "analyst"]

        templates = [
            "The {prof} finished the project before",
            "After the meeting, the {prof} explained that",
            "Everyone said the {prof} was tired because",
            "The {prof} returned to the office and",
            "When the alarm went off, the {prof} realized that",
            "Despite the late hour, the {prof} kept working because",
            "The {prof} answered the phone and said that",
            "On the way home, the {prof} thought about how",
        ]

        pronouns = [("she", "FEM", 1), ("he", "MASC", 0)]

        examples: list[Example] = []
        prof_lists = [(fem_skew, "FEM_SKEW", 1),
                      (masc_skew, "MASC_SKEW", 0),
                      (neutral, "NEUTRAL", 1)]
        for prof_list, _prof_label, ze in prof_lists:
            for prof in prof_list:
                for tpl in templates:
                    for pronoun, _pron_label, zc in pronouns:
                        # Build the prefix: the sentence ends right before the pronoun.
                        prefix = tpl.format(prof=prof)
                        examples.append(Example(sentence=prefix, zc=zc, ze=ze))
        rng.shuffle(examples)
        return examples


# ---------------------------------------------------------------------------
# SST-2 task
# ---------------------------------------------------------------------------

class SST2Task(Task):
    """SST-2 binary sentiment.

    Zc = sentiment label (1 positive / 0 negative).
    Ze = sentence length bucket (1 long / 0 short, split at training median).

    Sentence length is a deliberately non-sentiment feature -- a probe whose
    intervention removes Zc but also corrupts Ze is failing to be selective.

    Data: HuggingFace SST-2 (datasets library). One line per example:
        sentence\tlabel
    where label is "0" or "1". If file paths point to existing TSVs, we use
    those; otherwise we download via the datasets library.
    """

    name = "sst2"
    chance_accuracy = 0.5
    zc_description = "sentiment label (positive / negative)"
    ze_description = "sentence length bucket (long / short)"

    def load(self, paths, max_examples, seed):
        paths = [Path(p) for p in paths]
        if any(p.exists() for p in paths):
            rows = self._load_from_files(paths)
        else:
            rows = self._download_from_hub()
        rng = random.Random(seed)
        rng.shuffle(rows)
        if max_examples is not None:
            rows = rows[:max_examples]

        # Compute length-bucket split point on this batch.
        lengths = [len(s.split()) for s, _ in rows]
        if not lengths:
            return []
        lengths_sorted = sorted(lengths)
        median_len = lengths_sorted[len(lengths_sorted) // 2]

        examples: list[Example] = []
        for sent, label in rows:
            n = len(sent.split())
            ze = 1 if n > median_len else 0
            examples.append(Example(sentence=sent, zc=label, ze=ze))
        return examples

    def _load_from_files(self, paths):
        rows: list[tuple[str, int]] = []
        for path in paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if "\t" not in line:
                        continue
                    sent, lbl = line.rsplit("\t", 1)
                    if lbl in {"0", "1"}:
                        rows.append((sent, int(lbl)))
        return rows

    def _download_from_hub(self):
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "SST-2 download requires the `datasets` package. "
                "Install with: pip install datasets"
            ) from e
        ds = load_dataset("stanfordnlp/sst2", split="train")
        rows = [(row["sentence"].strip(), int(row["label"]))
                for row in ds if row["sentence"].strip()]
        return rows


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_TASK_REGISTRY: dict[str, type[Task]] = {
    "sva": SVATask,
    "gender": GenderTask,
    "sst2": SST2Task,
}


def get_task(name: str) -> Task:
    if name not in _TASK_REGISTRY:
        raise ValueError(f"Unknown task: {name!r}. "
                         f"Known: {sorted(_TASK_REGISTRY)}")
    return _TASK_REGISTRY[name]()
