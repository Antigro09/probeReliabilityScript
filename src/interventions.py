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
import torch.nn as nn
import torch.nn.functional as F

from .probes import LinearProbe


# ---------------------------------------------------------------------------
# INLP
# ---------------------------------------------------------------------------

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
    d = X.shape[1]
    Xc = X.detach().clone().float()
    P = torch.eye(d)
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
        # Project out the discriminant direction
        w = clf.linear.weight.detach().cpu()  # (2, D)
        v = F.normalize(w[0:1], dim=1)         # (1, D)
        P_step = torch.eye(d) - v.T @ v
        P = P @ P_step
        Xc = Xc @ P_step
    return P


def apply_inlp(X: torch.Tensor, zc: torch.Tensor, *,
               device: torch.device, num_iters: int = 10) -> torch.Tensor:
    """Convenience wrapper: compute P from (X, zc) and return X @ P."""
    P = inlp_projection(X, zc, device=device, num_iters=num_iters)
    return X @ P


# ---------------------------------------------------------------------------
# RLACE (rank-r adversarial erasure)
# ---------------------------------------------------------------------------

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
    d = X.shape[1]
    # Adversarial subspace U: D x rank, orthonormalized each step.
    # NOTE: must call requires_grad_() on a leaf tensor; multiplying first
    # would create a non-leaf which Adam can't optimize.
    U = (torch.randn(d, rank, device=device) * 0.01).requires_grad_(True)
    clf = nn.Linear(d, 2).to(device)

    opt_u = torch.optim.Adam([U], lr=lr)
    opt_c = torch.optim.Adam(clf.parameters(), lr=lr)

    Xd = X.to(device).float()
    yd = y.to(device)

    for _ in range(steps):
        # Inner loop: train classifier on projected X
        for _ in range(inner_steps):
            with torch.no_grad():
                Q, _ = torch.linalg.qr(U)  # orthonormalize
            P = torch.eye(d, device=device) - Q @ Q.T
            Xp = Xd @ P
            logits = clf(Xp.detach())
            loss_c = F.cross_entropy(logits, yd)
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()
        # Outer step: U tries to make classifier WORSE
        with torch.no_grad():
            Q, _ = torch.linalg.qr(U)
        P = torch.eye(d, device=device) - Q @ Q.T
        Xp = Xd @ P
        logits = clf(Xp)
        loss_u = -F.cross_entropy(logits, yd)
        opt_u.zero_grad()
        loss_u.backward()
        opt_u.step()

    with torch.no_grad():
        Q, _ = torch.linalg.qr(U)
        P = torch.eye(d, device=device) - Q @ Q.T
    return P.cpu()


def apply_rlace(X: torch.Tensor, zc: torch.Tensor, *,
                device: torch.device, rank: int = 1,
                steps: int = 500) -> torch.Tensor:
    P = rlace_projection(X, zc, device=device, rank=rank, steps=steps)
    return X @ P


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
    w = validation_probe.linear.weight.detach()    # (2, D)
    direction = (w[1] - w[0]).to(device)
    direction = direction / (direction.norm() + 1e-9)

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
