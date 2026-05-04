"""
Reproducibility utilities.

Main-track papers require deterministic results. This module provides:
    1. A single set_seed() that locks Python, NumPy, PyTorch CPU+CUDA RNGs.
    2. Deterministic CuDNN flags.
    3. Hash utilities for data provenance.

Note that full determinism in PyTorch costs ~10-20% throughput. We pay it.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Lock all RNGs. Call this at the top of every script entry point."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch >= 1.8 supports algorithm-level determinism.
        # warn_only=True because some ops (e.g. some attention kernels)
        # don't have deterministic implementations and would otherwise raise.
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Required for some CUDA operations to actually be deterministic.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def hash_examples(examples) -> str:
    """
    Stable hash of a list of LinzenExample objects. Used for cache keys
    and for the metadata sidecar to verify that two runs used the same data.
    """
    h = hashlib.sha256()
    for ex in examples:
        h.update(ex.sentence.encode("utf-8"))
        h.update(bytes([ex.zc, ex.ze]))
    return h.hexdigest()[:16]


def write_provenance(path: Path, metadata: dict) -> None:
    """Save metadata as JSON next to a tensor cache file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def read_provenance(path: Path) -> dict | None:
    """Read metadata if present."""
    path = Path(path)
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)
