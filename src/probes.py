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
    Manifold-approximated Kernel Alignment between two (N, N) kernels.

    Following Islam et al. 2025:
        MKA(K, L) = (<K, L> - D^2) / sqrt((<K, K> - D^2)(<L, L> - D^2))
    where D^2 = mean(K) * mean(L).

    Returns a scalar in roughly [-1, 1]. Higher means more aligned.
    """
    D2 = K.mean() * L.mean() * (K.numel())
    # Equivalent: D2 = (K.sum()/N**2) * (L.sum()/N**2) * N**2 = K.sum()*L.sum()/N**2
    # We compute it as in the original notebook for parity:
    n = K.numel()
    D2 = (K.sum() / n) * (L.sum() / n) * n
    num = (K * L).sum() - D2
    denom_sq = ((K * K).sum() - D2) * ((L * L).sum() - D2)
    den = torch.sqrt(torch.clamp(denom_sq, min=1e-12))
    return num / den


class MKAProbe(_ProbeBase):
    """
    MLP probe with Manifold Kernel Alignment regularization.

    During training, the regularization term encourages the probe's
    pre-output representation (post-ReLU hidden layer) to preserve the
    local neighborhood structure of the input representation.

    The regularizer is applied per-batch, computed via:
        L_mka = -mka_score(knn_kernel(input), knn_kernel(hidden))
    Negating because we want to MAXIMIZE alignment.
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
        """Compute -MKA(input_kNN, hidden_kNN)."""
        K_in = knn_kernel(x_input, k=self.knn_k)
        K_h = knn_kernel(hidden, k=self.knn_k)
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
