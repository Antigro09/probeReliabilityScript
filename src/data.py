"""
Linzen subject-verb agreement dataset loading + Ze computation.

The Linzen Number Prediction dataset format is:
    LABEL\tSENTENCE_PREFIX
where LABEL is VBZ (singular) or VBP (plural), and SENTENCE_PREFIX
ends right BEFORE the target verb. The probe predicts the number of
the next (omitted) verb from the prefix's representation.

Rare words are replaced inline with Penn Treebank POS tags (NN, NNS, NNP,
NNPS, JJ, VBD, etc.); common words remain as plain English. To compute
the grammatical number of the last noun, we run spaCy on the prefix and
override its POS for any token that is itself an inline tag.

Zc = grammatical number of the target verb's subject (= the label).
Ze = grammatical number of the last noun in the prefix. This is the
     "attractor" used to test whether interventions selectively modify Zc
     without disturbing other grammatical features.

Convention: 1 = singular, 0 = plural.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# spaCy is loaded lazily because it's slow to initialize. Stanza is avoided
# here because (1) it's heavier, (2) it's slower on Windows, and (3) for this
# dataset the inline POS tags carry most of the signal we need anyway.
_NLP = None


# Penn Treebank tag sets ------------------------------------------------------

SINGULAR_NOUN_TAGS = {"NN", "NNP"}
PLURAL_NOUN_TAGS = {"NNS", "NNPS"}
NOUN_TAGS = SINGULAR_NOUN_TAGS | PLURAL_NOUN_TAGS

# These are the literal tokens the dataset uses to replace rare words.
INLINE_TAG_TOKENS = NOUN_TAGS | {"JJ", "JJR", "JJS", "VBD", "VBN", "VBG",
                                 "RB", "CD", "UH", "FW", "SYM"}


@dataclass
class LinzenExample:
    """One example from the Linzen number prediction dataset."""
    sentence: str       # the prefix (ends right before the target verb)
    zc: int             # 1 = singular (VBZ), 0 = plural (VBP)
    ze: int             # 1 = singular last noun, 0 = plural last noun

    def to_dict(self) -> dict:
        return {"sentence": self.sentence, "zc": self.zc, "ze": self.ze}


# Loading --------------------------------------------------------------------

def load_raw(paths: Iterable[Path]) -> list[tuple[str, str]]:
    """Load (label, prefix) pairs from one or more Linzen files."""
    rows: list[tuple[str, str]] = []
    for path in paths:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or "\t" not in line:
                    continue
                label, sentence = line.split("\t", 1)
                if label not in {"VBZ", "VBP"}:
                    continue
                rows.append((label, sentence))
    return rows


# Ze computation -------------------------------------------------------------

def _get_spacy():
    """Lazy-load spaCy. Required for Ze on common-word nouns."""
    global _NLP
    if _NLP is None:
        import spacy
        try:
            _NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        except OSError as e:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is not installed. "
                "Run: python -m spacy download en_core_web_sm"
            ) from e
    return _NLP


def _inline_tag_number(tok_text: str) -> int | None:
    """If the token is an inline POS tag for a noun, return its number.
    Returns None if the token is not a noun-tag."""
    if tok_text in SINGULAR_NOUN_TAGS:
        return 1
    if tok_text in PLURAL_NOUN_TAGS:
        return 0
    return None


def compute_ze(prefix: str) -> int:
    """
    Ze = grammatical number of the LAST noun in the prefix.

    Algorithm:
        1. Run spaCy on the full prefix (provides POS + Number features).
        2. Walk tokens right-to-left.
        3. For each token:
            a. If it's an inline noun tag (NN/NNP/NNS/NNPS), use the tag.
            b. If it's any other inline tag (JJ, VBD, ...), skip.
            c. Otherwise trust spaCy: if it's a NOUN/PROPN, return its
               number; else continue.
        4. If no noun is found, default to singular (1).

    Returns:
        1 for singular, 0 for plural.
    """
    nlp = _get_spacy()
    doc = nlp(prefix)
    return _ze_from_doc(doc)


def _ze_from_doc(doc) -> int:
    """Compute Ze given an already-parsed spaCy doc."""
    for tok in reversed(list(doc)):
        text = tok.text
        n = _inline_tag_number(text)
        if n is not None:
            return n
        if text in INLINE_TAG_TOKENS:
            continue
        if tok.pos_ in {"NOUN", "PROPN"}:
            morph = str(tok.morph) if tok.morph else ""
            return 0 if "Number=Plur" in morph else 1
    return 1


def compute_ze_batched(prefixes: list[str], batch_size: int = 256) -> list[int]:
    """
    Batched Ze computation using spaCy's nlp.pipe.
    Roughly 30-50x faster than calling compute_ze in a loop on 100K examples.
    """
    nlp = _get_spacy()
    out: list[int] = []
    for doc in nlp.pipe(prefixes, batch_size=batch_size):
        out.append(_ze_from_doc(doc))
    return out


# Splitting ------------------------------------------------------------------

def label_to_zc(label: str) -> int:
    """VBZ -> 1 (singular), VBP -> 0 (plural)."""
    return 1 if label == "VBZ" else 0


def build_examples(
    paths: Iterable[Path],
    max_examples: int | None = None,
    seed: int = 42,
    spacy_batch_size: int = 256,
) -> list[LinzenExample]:
    """Load raw rows, compute Ze in batch, return shuffled examples."""
    raw = load_raw(paths)
    rng = random.Random(seed)
    rng.shuffle(raw)
    if max_examples is not None:
        raw = raw[:max_examples]
    sentences = [s for _, s in raw]
    ze_vals = compute_ze_batched(sentences, batch_size=spacy_batch_size)
    examples: list[LinzenExample] = []
    for (label, sentence), ze in zip(raw, ze_vals):
        examples.append(
            LinzenExample(
                sentence=sentence,
                zc=label_to_zc(label),
                ze=ze,
            )
        )
    return examples


def split_balanced(
    examples: list[LinzenExample],
    val_frac: float = 0.4,
    inter_frac: float = 0.4,
    seed: int = 42,
) -> tuple[list[LinzenExample], list[LinzenExample], list[LinzenExample]]:
    """
    Split examples into (probe-train, intervention-train, test) splits while
    balancing across the four (zc, ze) cells.

    The remaining fraction (1 - val_frac - inter_frac) becomes the test set.
    """
    if val_frac + inter_frac >= 1.0:
        raise ValueError("val_frac + inter_frac must be < 1.0")
    rng = random.Random(seed)
    buckets: dict[tuple[int, int], list[LinzenExample]] = {
        (0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []
    }
    for ex in examples:
        buckets[(ex.zc, ex.ze)].append(ex)

    # Down-sample to the smallest bucket for balance.
    min_n = min(len(v) for v in buckets.values())
    balanced: list[LinzenExample] = []
    for v in buckets.values():
        rng.shuffle(v)
        balanced.extend(v[:min_n])
    rng.shuffle(balanced)

    n = len(balanced)
    n_val = int(val_frac * n)
    n_int = int(inter_frac * n)
    return (
        balanced[:n_val],
        balanced[n_val:n_val + n_int],
        balanced[n_val + n_int:],
    )
