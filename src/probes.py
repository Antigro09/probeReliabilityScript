"""
Probe architectures: Linear, MLP, and MKA-regularized MLP.

All probes:
    - Take a (B, D) input tensor
    - Output (B, num_classes) logits
    - Expose .flat_params() returning a single 1D tensor of all trainable
      parameters concatenated. This is used for Hessian directional alignment.

The MKAProbe applies an additional Manifold Kernel Alignment regularizer
during training, which pushes the probe-induced representation to preserve
the input's local neighborhood structure. The regularization weight is
controlled by mka_lambda; setting it to 0 makes MKAProbe behave identically
to MLPProbe (useful as an ablation).

v2 note: the regularizer's hidden-side kernel is a soft RBF kernel
(rbf_kernel), not a binary kNN adjacency. The binary kernel used in v1 was
built under torch.no_grad() from scatter_ of constants, so it was
piecewise-constant with zero gradient almost everywhere — the regularizer
added a constant to the loss and MKA collapsed to MLP (byte-identical
parameters in 1800/1800 seed-matched v1 records). Acceptance tests for the
fix live in tests/test_mka.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Base probe protocol
# ---------------------------------------------------------------------------

class _ProbeBase(nn.Module):
    """Base class. All probes implement flat_params()."""

    def flat_params(self) -> torch.Tensor:
        """Concatenate all trainable parameters into one 1D vector.
        Used by hessian.py for directional alignment with eigenvectors."""
        return torch.cat([p.detach().flatten() for p in self.parameters()
                          if p.requires_grad])

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

class LinearProbe(_ProbeBase):
    """Single linear layer. Used as the low-capacity baseline."""

    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        logits = self.linear(x)
        if return_hidden:
            return logits, x
        return logits


# ---------------------------------------------------------------------------
# MLP probe
# ---------------------------------------------------------------------------

class MLPProbe(_ProbeBase):
    """Two-layer MLP with ReLU. Higher-capacity probe."""

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)
        if return_hidden:
            return logits, h
        return logits


# ---------------------------------------------------------------------------
# MKA: Manifold Kernel Alignment
# ---------------------------------------------------------------------------

def pairwise_sq_dists(x: torch.Tensor) -> torch.Tensor:
    """
    (N, N) matrix of squared Euclidean distances, differentiable everywhere.

    Computed as ||xi||^2 + ||xj||^2 - 2<xi, xj> rather than via torch.cdist,
    whose backward is undefined at zero distance (the diagonal).
    """
    sq = (x * x).sum(dim=1, keepdim=True)          # (N, 1)
    d2 = sq + sq.T - 2.0 * (x @ x.T)
    return d2.clamp_min(0.0)


def rbf_kernel(x: torch.Tensor, k: int | None = None,
               bandwidth: float | torch.Tensor | None = None,
               zero_diagonal: bool = True) -> torch.Tensor:
    """
    Soft (differentiable) neighborhood kernel: K[i, j] = exp(-d2_ij / b).

    Bandwidth selection (in priority order):
        - explicit `bandwidth` if given;
        - if `k` is given: the median over points of the squared distance
          to the k-th nearest neighbor. This ties the soft kernel's
          resolution to the same k as the hard kNN kernel it is aligned
          against: a typical k-th neighbor gets weight e^-1, closer
          neighbors more, farther points decay smoothly. Without it, plain
          median-of-all-pairs bandwidth goes nearly flat in high dimension
          (distance concentration) and barely tracks neighborhood structure.
        - otherwise: median of all off-diagonal squared distances.

    The bandwidth is treated as a constant (detached), so gradients flow
    only through the distances — that is the gradient path the MKA
    regularizer needs.

    The diagonal is zeroed by default for consistency with knn_kernel,
    which also has a zero diagonal (self is excluded from the neighbor set).
    """
    d2 = pairwise_sq_dists(x)
    n = x.shape[0]
    if bandwidth is None:
        with torch.no_grad():
            if n < 2:
                bandwidth = torch.tensor(1.0, device=x.device)
            elif k is not None:
                k_actual = max(1, min(k, n - 1))
                d2_noself = d2 + torch.eye(n, device=x.device) * torch.finfo(d2.dtype).max / 2
                kth, _ = torch.topk(d2_noself, k=k_actual, dim=1, largest=False)
                bandwidth = kth[:, -1].median().clamp_min(1e-12)
            else:
                off_diag = d2[~torch.eye(n, dtype=torch.bool, device=x.device)]
                bandwidth = off_diag.median().clamp_min(1e-12)
    K = torch.exp(-d2 / bandwidth)
    if zero_diagonal:
        K = K * (1.0 - torch.eye(n, device=x.device, dtype=K.dtype))
    return K


def knn_kernel(x: torch.Tensor, k: int = 10) -> torch.Tensor:
    """
    Binary k-nearest-neighbor adjacency matrix for x of shape (N, D).
    Returns an (N, N) matrix where K[i, j] = 1 iff j is among i's k
    nearest neighbors (excluding self), else 0.

    Note: this is asymmetric in general. We use it directly without
    symmetrization to match the original MKA paper.
    """
    n = x.shape[0]
    k_actual = min(k, n - 1)
    if k_actual < 1:
        return torch.zeros(n, n, device=x.device)
    with torch.no_grad():
        dist = torch.cdist(x, x, p=2)
        _, knn_idx = torch.topk(dist, k=k_actual + 1, largest=False)
        knn_idx = knn_idx[:, 1:]  # exclude self
        K = torch.zeros(n, n, device=x.device)
        K.scatter_(1, knn_idx, 1.0)
    return K


def mka_score(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    Manifold-approximated Kernel Alignment between two (N, N) kernels:
    the cosine similarity of the mean-centered kernels,

        MKA(K, L) = <K - mean(K), L - mean(L)>
                    / (||K - mean(K)|| * ||L - mean(L)||)

    Bounded in [-1, 1] by Cauchy-Schwarz. Higher means more aligned.

    v2 fix: the v1 implementation subtracted the CROSS term
    mean(K)*mean(L)*N^2 inside BOTH denominator variance terms. That is
    algebraically equal to this form when mean(K) == mean(L) — always true
    in v1, where both kernels were binary kNN adjacencies with identical
    row sums — but for kernels with different means a denominator term can
    go negative, hit the clamp, and blow the score up by ~1e6. Each
    variance term now uses its own kernel's mean, which is the standard
    centered-alignment definition and is non-negative by construction.
    """
    n = K.numel()
    K_mean = K.mean()
    L_mean = L.mean()
    num = (K * L).sum() - n * K_mean * L_mean
    var_K = (K * K).sum() - n * K_mean * K_mean
    var_L = (L * L).sum() - n * L_mean * L_mean
    den = torch.sqrt(torch.clamp(var_K, min=1e-12) *
                     torch.clamp(var_L, min=1e-12))
    return num / den


class MKAProbe(_ProbeBase):
    """
    MLP probe with Manifold Kernel Alignment regularization.

    During training, the regularization term encourages the probe's
    pre-output representation (post-ReLU hidden layer) to preserve the
    local neighborhood structure of the input representation.

    The regularizer is applied per-batch, computed via:
        L_mka = -mka_score(knn_kernel(input), rbf_kernel(hidden))
    Negating because we want to MAXIMIZE alignment.

    Kernel choice (v2 fix): the HIDDEN kernel must be differentiable
    w.r.t. the activations, otherwise the regularizer contributes zero
    gradient and MKA training is byte-identical to plain MLP training —
    which is exactly what happened in v1, where both kernels were binary
    kNN adjacencies built under torch.no_grad(). The hidden side now uses
    a soft RBF kernel with median-heuristic bandwidth. The INPUT kernel
    stays hard binary kNN: there is no gradient path through the input
    anyway, and a hard neighbor set is a cleaner alignment target.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 num_classes: int = 2, mka_lambda: float = 0.1,
                 knn_k: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.mka_lambda = mka_lambda
        self.knn_k = knn_k

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)
        if return_hidden:
            return logits, h
        return logits

    def mka_loss(self, x_input: torch.Tensor,
                 hidden: torch.Tensor) -> torch.Tensor:
        """Compute -MKA(input_kNN, hidden_RBF). Differentiable in `hidden`."""
        K_in = knn_kernel(x_input, k=self.knn_k)
        K_h = rbf_kernel(hidden, k=self.knn_k)
        return -mka_score(K_in, K_h)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class ProbeTrainConfig:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 256
    mka_lambda: float | None = None  # if set, override probe's lambda


def train_probe(
    probe: _ProbeBase,
    X: torch.Tensor,
    y: torch.Tensor,
    cfg: ProbeTrainConfig,
    device: torch.device,
    verbose: bool = False,
) -> _ProbeBase:
    """
    Train a probe with mini-batch AdamW. Handles the MKA regularizer
    automatically if probe is an MKAProbe.

    Returns the probe in eval mode after training. The probe is modified
    in place; the return value is for chaining.
    """
    probe.to(device).train()
    opt = torch.optim.AdamW(probe.parameters(),
                            lr=cfg.lr, weight_decay=cfg.weight_decay)
    n = X.shape[0]
    is_mka = isinstance(probe, MKAProbe)

    for epoch in range(cfg.epochs):
        perm = torch.randperm(n, device=X.device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            xb = X[idx].to(device).float()
            yb = y[idx].to(device)
            logits, hidden = probe(xb, return_hidden=True)
            ce = F.cross_entropy(logits, yb)
            if is_mka:
                lam = cfg.mka_lambda if cfg.mka_lambda is not None else probe.mka_lambda
                if lam > 0:
                    mka_term = probe.mka_loss(xb, hidden)
                    loss = ce + lam * mka_term
                else:
                    loss = ce
            else:
                loss = ce
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        if verbose and (epoch + 1) % max(1, cfg.epochs // 5) == 0:
            print(f"    epoch {epoch+1:3d}/{cfg.epochs}  loss={epoch_loss/n_batches:.4f}")

    probe.eval()
    return probe


@torch.no_grad()
def probe_accuracy(probe: _ProbeBase, X: torch.Tensor, y: torch.Tensor,
                   device: torch.device, batch_size: int = 1024) -> float:
    """Compute classification accuracy of a probe on (X, y)."""
    probe.eval()
    n = X.shape[0]
    correct = 0
    for start in range(0, n, batch_size):
        xb = X[start:start + batch_size].to(device).float()
        yb = y[start:start + batch_size].to(device)
        preds = probe(xb).argmax(dim=-1)
        correct += (preds == yb).sum().item()
    return correct / n


@torch.no_grad()
def probe_logits(probe: _ProbeBase, X: torch.Tensor,
                 device: torch.device, batch_size: int = 1024) -> torch.Tensor:
    """Return (N, C) logits for X. Useful for intervention metrics."""
    probe.eval()
    out = []
    for start in range(0, X.shape[0], batch_size):
        xb = X[start:start + batch_size].to(device).float()
        out.append(probe(xb).cpu())
    return torch.cat(out, dim=0)
