"""
Five causal interventions on representation space:
    INLP    — Iterative Null-space Projection (Ravfogel et al. 2020)
    RLACE   — Relaxed Linear Adversarial Concept Erasure (Ravfogel et al. 2022)
    AlterRep — Counterfactual representation perturbation (Ravfogel et al. 2021)
    FGSM    — Fast Gradient Sign Method (Goodfellow et al. 2015)
    PGD     — Projected Gradient Descent (Madry et al. 2018)

All interventions share the same interface:
    apply(X, zc, ze, validation_probe, ...) -> X_post
where X is (N, D) representations and X_post is the modified (N, D) tensor.

INLP / RLACE produce a fixed projection that's applied uniformly to all
inputs (population-level). AlterRep / FGSM / PGD modify each input
individually based on its own gradient. We expose both styles through
the same callable API.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .probes import LinearProbe


def _binary_linear_direction(probe: LinearProbe) -> torch.Tensor:
    """Unit vector for the binary logit difference ``logit_1-logit_0``."""
    weight = probe.linear.weight.detach()
    direction = weight[1] - weight[0]
    return direction / (direction.norm() + 1e-9)


def _apply_orthogonal_erasure(
    X: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """Apply ``I - Q Q^T`` without materializing the square projector.

    ``basis`` must contain orthonormal columns.  The low-rank expression uses
    ``O(NDr)`` work and ``O(Nr)`` temporary storage instead of the ``O(ND^2)``
    dense multiplication required by ``X @ (I - Q @ Q.T)``.
    """

    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional tensor")
    if basis.ndim != 2:
        raise ValueError("basis must be a two-dimensional tensor")
    if X.shape[1] != basis.shape[0]:
        raise ValueError("basis dimension does not match X")
    return (X - (X @ basis) @ basis.T).contiguous()


# ---------------------------------------------------------------------------
# INLP
# ---------------------------------------------------------------------------

def _fit_inlp_subspace(
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    num_iters: int = 10,
    epochs: int = 100,
    lr: float = 0.1,
    early_stop_acc: float = 0.55,
) -> torch.Tensor:
    """Fit and return an orthonormal basis for the INLP row space."""

    d = X.shape[1]
    Xc = X.detach().cpu().clone().float()
    basis = torch.empty((d, 0), dtype=Xc.dtype)
    for _ in range(num_iters):
        clf = LinearProbe(d).to(device)
        opt = torch.optim.SGD(clf.parameters(), lr=lr)
        Xd = Xc.to(device)
        yd = y.to(device)
        for _ in range(epochs):
            logits = clf(Xd)
            loss = F.cross_entropy(logits, yd)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            preds = clf(Xd).argmax(dim=-1)
            acc = (preds == yd).float().mean().item()
        if acc < early_stop_acc:
            break

        # A two-logit softmax depends on (w[1] - w[0])^T x.  Remove any
        # component already represented in the accumulated row space so that
        # the returned basis defines a genuine orthogonal projector.
        direction = _binary_linear_direction(clf).cpu().float()
        if basis.shape[1]:
            direction = direction - basis @ (basis.T @ direction)
        direction_norm = direction.norm()
        if not torch.isfinite(direction_norm) or float(direction_norm) <= 1.0e-8:
            break
        new_direction = (direction / direction_norm).unsqueeze(1)
        basis = torch.cat((basis, new_direction), dim=1)
        Xc = _apply_orthogonal_erasure(Xc, new_direction)
    return basis


def inlp_projection(
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    num_iters: int = 10,
    epochs: int = 100,
    lr: float = 0.1,
    early_stop_acc: float = 0.55,
) -> torch.Tensor:
    """
    Compute the INLP projection matrix P that iteratively nullifies
    linearly decodable target information.

    Returns:
        P: (D, D) tensor on CPU. To intervene: X_post = X @ P.
    """
    basis = _fit_inlp_subspace(
        X,
        y,
        device=device,
        num_iters=num_iters,
        epochs=epochs,
        lr=lr,
        early_stop_acc=early_stop_acc,
    )
    d = X.shape[1]
    return torch.eye(d, dtype=basis.dtype) - basis @ basis.T


def apply_inlp(X: torch.Tensor, zc: torch.Tensor, *,
               device: torch.device, num_iters: int = 10) -> torch.Tensor:
    """Fit INLP and apply its low-rank erasure without constructing ``P``."""
    basis = _fit_inlp_subspace(X, zc, device=device, num_iters=num_iters)
    Xd = X.detach().cpu().float()
    return _apply_orthogonal_erasure(Xd, basis)


# ---------------------------------------------------------------------------
# RLACE (rank-r adversarial erasure)
# ---------------------------------------------------------------------------

def _fit_rlace_subspace(
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    rank: int = 1,
    steps: int = 500,
    lr: float = 1e-2,
    inner_steps: int = 5,
) -> torch.Tensor:
    """Fit the approximate-RLACE adversarial subspace on ``device``."""

    d = X.shape[1]
    if rank < 1 or rank > d:
        raise ValueError(f"rank must be between 1 and {d}, got {rank}")

    # U remains the optimizer leaf.  QR is differentiable in the outer step,
    # while the inner classifier step intentionally treats the current
    # projection as fixed.
    U = (torch.randn(d, rank, device=device) * 0.01).requires_grad_(True)
    clf = nn.Linear(d, 2).to(device)
    opt_u = torch.optim.Adam([U], lr=lr)
    opt_c = torch.optim.Adam(clf.parameters(), lr=lr)

    Xd = X.to(device).float()
    yd = y.to(device)

    for _ in range(steps):
        for _ in range(inner_steps):
            with torch.no_grad():
                fixed_basis = torch.linalg.qr(U, mode="reduced")[0]
            projected = _apply_orthogonal_erasure(Xd, fixed_basis)
            logits = clf(projected.detach())
            loss_c = F.cross_entropy(logits, yd)
            opt_c.zero_grad(set_to_none=True)
            loss_c.backward()
            opt_c.step()

        # Keep QR in the autograd graph: this is the only path from the outer
        # adversarial loss back to U.  Detaching the classifier weights avoids
        # accumulating irrelevant classifier gradients while retaining the
        # derivative through its input.
        basis = torch.linalg.qr(U, mode="reduced")[0]
        projected = _apply_orthogonal_erasure(Xd, basis)
        logits = F.linear(projected, clf.weight.detach(), clf.bias.detach())
        loss_u = -F.cross_entropy(logits, yd)
        opt_u.zero_grad(set_to_none=True)
        loss_u.backward()
        if U.grad is None:
            raise RuntimeError("RLACE adversarial subspace received no gradient")
        if not torch.isfinite(U.grad).all():
            raise FloatingPointError("RLACE adversarial subspace gradient is non-finite")
        opt_u.step()

        # A well-conditioned full-rank representative makes the next QR and
        # its derivative stable without changing the represented subspace.
        with torch.no_grad():
            U.copy_(torch.linalg.qr(U, mode="reduced")[0])

    with torch.no_grad():
        basis = torch.linalg.qr(U, mode="reduced")[0]
    if not torch.isfinite(basis).all():
        raise FloatingPointError("RLACE produced a non-finite subspace")
    return basis


def rlace_projection(
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    rank: int = 1,
    steps: int = 500,
    lr: float = 1e-2,
    inner_steps: int = 5,
) -> torch.Tensor:
    """
    Approximate RLACE: solve a min-max where a classifier tries to predict
    y from a rank-r-projected X, while we adversarially shrink the
    classifier's accuracy by adjusting the projection.

    This is a pragmatic, simplified RLACE — sufficient for our diagnostic
    purposes. Returns the projection matrix P = I - U U^T where U is the
    rank-r adversarial subspace.
    """
    basis = _fit_rlace_subspace(
        X,
        y,
        device=device,
        rank=rank,
        steps=steps,
        lr=lr,
        inner_steps=inner_steps,
    )
    d = X.shape[1]
    with torch.no_grad():
        projection = (
            torch.eye(d, device=device, dtype=basis.dtype) - basis @ basis.T
        )
    return projection.cpu()


def apply_rlace(X: torch.Tensor, zc: torch.Tensor, *,
                device: torch.device, rank: int = 1,
                steps: int = 500) -> torch.Tensor:
    basis = _fit_rlace_subspace(
        X, zc, device=device, rank=rank, steps=steps
    )
    Xd = X.to(device).float()
    return _apply_orthogonal_erasure(Xd, basis).cpu()


# ---------------------------------------------------------------------------
# AlterRep — direction-based counterfactual
# ---------------------------------------------------------------------------

def apply_alterrep(
    X: torch.Tensor,
    zc: torch.Tensor,
    *,
    validation_probe: LinearProbe,
    device: torch.device,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    AlterRep: shift each example along the validation probe's
    discriminant direction in the direction OPPOSITE its current label.

    For a binary linear probe with weight w, the discriminant direction
    is w[1] - w[0]. We shift each example by alpha * (sign-flip * unit_dir).
    """
    direction = _binary_linear_direction(validation_probe).to(device)

    Xd = X.to(device).float()
    zd = zc.to(device)
    # If label is 1 (singular), shift toward 0 (plural) means subtract direction.
    sign = torch.where(zd == 1, -1.0, 1.0).unsqueeze(1)  # (N, 1)
    Xp = Xd + alpha * sign * direction.unsqueeze(0)
    return Xp.cpu()


# ---------------------------------------------------------------------------
# FGSM and PGD — adversarial
# ---------------------------------------------------------------------------

def apply_fgsm(
    X: torch.Tensor,
    zc: torch.Tensor,
    *,
    validation_probe: LinearProbe,
    device: torch.device,
    epsilon: float = 0.5,
) -> torch.Tensor:
    """
    Fast Gradient Sign Method: perturb each example by epsilon * sign(grad)
    where grad is the gradient of the cross-entropy loss w.r.t. the
    representation, evaluated at the validation probe.
    """
    Xd = X.to(device).float().clone().requires_grad_(True)
    yd = zc.to(device)
    logits = validation_probe(Xd)
    loss = F.cross_entropy(logits, yd)
    grad = torch.autograd.grad(loss, Xd)[0]
    Xp = Xd.detach() + epsilon * grad.sign()
    return Xp.cpu()


def apply_pgd(
    X: torch.Tensor,
    zc: torch.Tensor,
    *,
    validation_probe: LinearProbe,
    device: torch.device,
    epsilon: float = 0.5,
    steps: int = 10,
    alpha: float = 0.1,
) -> torch.Tensor:
    """
    Projected Gradient Descent: iterative FGSM clipped to an L_inf ball
    of radius epsilon around the original point.
    """
    Xd = X.to(device).float()
    yd = zc.to(device)
    Xp = Xd.clone()
    for _ in range(steps):
        Xp = Xp.detach().requires_grad_(True)
        logits = validation_probe(Xp)
        loss = F.cross_entropy(logits, yd)
        grad = torch.autograd.grad(loss, Xp)[0]
        Xp = Xp.detach() + alpha * grad.sign()
        # Project back to L_inf ball of radius epsilon
        delta = torch.clamp(Xp - Xd, min=-epsilon, max=epsilon)
        Xp = Xd + delta
    return Xp.cpu()


# ---------------------------------------------------------------------------
# Unified registry
# ---------------------------------------------------------------------------

@dataclass
class InterventionConfig:
    inlp_iters: int = 10
    rlace_rank: int = 1
    rlace_steps: int = 500
    alterrep_alpha: float = 1.0
    fgsm_eps: float = 0.5
    pgd_eps: float = 0.5
    pgd_steps: int = 10
    pgd_alpha: float = 0.1


def run_all_interventions(
    X: torch.Tensor,
    zc: torch.Tensor,
    validation_probe: LinearProbe,
    device: torch.device,
    cfg: InterventionConfig | None = None,
) -> dict[str, torch.Tensor]:
    """
    Apply all five interventions and return a dict
        {"INLP": X_post, "RLACE": X_post, ...}
    """
    cfg = cfg or InterventionConfig()
    out: dict[str, torch.Tensor] = {}
    out["INLP"] = apply_inlp(X, zc, device=device, num_iters=cfg.inlp_iters)
    out["RLACE"] = apply_rlace(X, zc, device=device,
                               rank=cfg.rlace_rank, steps=cfg.rlace_steps)
    out["AlterRep"] = apply_alterrep(X, zc, validation_probe=validation_probe,
                                     device=device, alpha=cfg.alterrep_alpha)
    out["FGSM"] = apply_fgsm(X, zc, validation_probe=validation_probe,
                             device=device, epsilon=cfg.fgsm_eps)
    out["PGD"] = apply_pgd(X, zc, validation_probe=validation_probe,
                           device=device, epsilon=cfg.pgd_eps,
                           steps=cfg.pgd_steps, alpha=cfg.pgd_alpha)
    return out
