"""
Hessian eigenspectrum and directional alignment of the probe loss.

This module implements the paper's central diagnostic: the alignment
between the probe's parameter vector and the high-curvature eigenvectors
of its loss Hessian.

CRITICAL FIX OVER ORIGINAL NOTEBOOK CODE:
    The original implementation extracted only the *output layer* weight,
    flattened it, and compared it against a slice of the eigenvector. This
    silently misaligned shapes for non-linear probes — e.g. for an MLP, the
    code took the first |fc2.weight| entries of the eigenvector, which
    actually correspond to a chunk of fc1.weight. The reported "alignment"
    numbers in the paper were therefore measuring something incoherent for
    MLP and MKA probes.

    The correct approach: flatten ALL trainable probe parameters into one
    vector, flatten ALL eigenvector components into one vector of the same
    length, then compute cosine similarity. Both vectors live in the same
    parameter space, so this is the only well-defined alignment.

We use PyHessian for power-iteration eigenvalue/eigenvector estimation.
For probes (≪ 1M params), this is fast and accurate.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F

try:
    from pyhessian import hessian as pyhessian_hessian
except ImportError as e:
    raise ImportError(
        "pyhessian is required. Install with: pip install pyhessian"
    ) from e

from .probes import _ProbeBase


# ---------------------------------------------------------------------------
# Eigenspectrum estimation
# ---------------------------------------------------------------------------

@dataclass
class HessianSpectrum:
    """
    Output of Hessian analysis at one (probe, data) point.

    eigenvalues_top:  list of length top_n, descending
    eigenvalues_bot:  list of length bottom_n, ascending (smallest first)
    eigvecs_top:      list of length top_n; each item is a list of tensors
                      (one tensor per parameter) - PyHessian's native format
    eigvecs_bot:      same as eigvecs_top, for the smallest eigenvalues
    align_top:        list of length top_n - cosine sim with probe params
    align_bot:        list of length bottom_n
    """
    eigenvalues_top: list[float]
    eigenvalues_bot: list[float]
    eigvecs_top: list
    eigvecs_bot: list
    align_top: list[float]
    align_bot: list[float]

    @property
    def lambda_max(self) -> float:
        return float(self.eigenvalues_top[0]) if self.eigenvalues_top else 0.0

    @property
    def lambda_min(self) -> float:
        return float(self.eigenvalues_bot[0]) if self.eigenvalues_bot else 0.0

    @property
    def mean_align_top(self) -> float:
        return sum(self.align_top) / max(1, len(self.align_top))

    @property
    def mean_align_bot(self) -> float:
        return sum(self.align_bot) / max(1, len(self.align_bot))

    def as_dict(self) -> dict:
        return {
            "lambda_max": self.lambda_max,
            "lambda_min": self.lambda_min,
            "mean_align_top": self.mean_align_top,
            "mean_align_bot": self.mean_align_bot,
            "eigenvalues_top": [float(e) for e in self.eigenvalues_top],
            "eigenvalues_bot": [float(e) for e in self.eigenvalues_bot],
            "align_top": [float(a) for a in self.align_top],
            "align_bot": [float(a) for a in self.align_bot],
        }


def _eigvec_to_flat(eigvec, device: torch.device | None = None) -> torch.Tensor:
    """
    PyHessian returns each eigenvector as a list of tensors mirroring the
    parameter list. Flatten and concatenate to a single 1D vector.
    """
    if isinstance(eigvec, list):
        parts = [v.detach().flatten().cpu() for v in eigvec]
        return torch.cat(parts)
    return eigvec.detach().flatten().cpu()


def directional_alignment(
    probe: _ProbeBase,
    eigvecs: list,
) -> list[float]:
    """
    Compute |cos(probe_params, v)| for each eigenvector v.

    The probe's full flat parameter vector is compared against each
    eigenvector flattened in the SAME parameter ordering (PyHessian uses
    the same ordering as `model.parameters()`, which is what _ProbeBase
    flat_params() uses).
    """
    w = probe.flat_params().cpu()
    w_norm = F.normalize(w, dim=0)
    out: list[float] = []
    for v in eigvecs:
        v_flat = _eigvec_to_flat(v)
        if v_flat.shape != w_norm.shape:
            raise RuntimeError(
                f"Shape mismatch in directional_alignment: probe has "
                f"{w_norm.shape[0]} params but eigenvector has "
                f"{v_flat.shape[0]} entries. This indicates the eigenvector "
                f"was computed against a different parameter set than the "
                f"probe currently holds."
            )
        v_norm = F.normalize(v_flat, dim=0)
        out.append(float(torch.dot(w_norm, v_norm).abs().item()))
    return out


def compute_hessian_spectrum(
    probe: _ProbeBase,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    top_n: int = 20,
    bottom_n: int = 20,
    max_iter: int = 100,
    tol: float = 1e-3,
) -> HessianSpectrum:
    """
    Compute the top-n and bottom-n eigenvalues and eigenvectors of the
    probe's training loss Hessian, plus their alignment with the probe's
    current parameter vector.

    Implementation notes:
        - We deepcopy the probe to avoid modifying its parameters during
          PyHessian's power iteration.
        - PyHessian uses cross_entropy by default, matching our probes.
        - The `bottom_n` eigenvalues are obtained by negating the loss and
          re-running power iteration (PyHessian doesn't expose this directly,
          so we wrap with a flipped-sign criterion).
    """
    probe_copy = copy.deepcopy(probe).to(device).eval()

    Xd = X.to(device).float()
    yd = y.to(device)

    use_cuda = (device.type == "cuda")

    # --- Top-n eigenvalues ---
    hess_top = pyhessian_hessian(
        probe_copy, F.cross_entropy,
        data=(Xd, yd), cuda=use_cuda,
    )
    eigvals_top, eigvecs_top = hess_top.eigenvalues(
        maxIter=max_iter, tol=tol, top_n=top_n,
    )

    align_top = directional_alignment(probe, eigvecs_top)

    # --- Bottom-n eigenvalues via negated loss ---
    if bottom_n > 0:
        def neg_ce(logits, targets):
            return -F.cross_entropy(logits, targets)

        probe_copy_bot = copy.deepcopy(probe).to(device).eval()
        hess_bot = pyhessian_hessian(
            probe_copy_bot, neg_ce,
            data=(Xd, yd), cuda=use_cuda,
        )
        eigvals_bot_neg, eigvecs_bot = hess_bot.eigenvalues(
            maxIter=max_iter, tol=tol, top_n=bottom_n,
        )
        # Negate back: largest eigenvalues of -L = -smallest eigenvalues of L
        eigvals_bot = [-v for v in eigvals_bot_neg]
        align_bot = directional_alignment(probe, eigvecs_bot)
    else:
        eigvals_bot = []
        eigvecs_bot = []
        align_bot = []

    return HessianSpectrum(
        eigenvalues_top=list(eigvals_top),
        eigenvalues_bot=eigvals_bot,
        eigvecs_top=eigvecs_top,
        eigvecs_bot=eigvecs_bot,
        align_top=align_top,
        align_bot=align_bot,
    )
