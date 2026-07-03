"""
Learnability gate (WS2): refuse to launch a benchmark on an unlearnable task.

The v1 gender task shipped with sentences truncated BEFORE the pronoun, so
the target label was not a function of the input and every probe sat at
chance — and 900 records per model were burned before anyone noticed. This
module turns that postmortem into a launch check.

For each feature (Zc and Ze), at a designated layer's representations:

    1. FLOOR: a linear probe trained on half the examples must reach a
       registered accuracy floor on the held-out half. If the feature is
       not linearly decodable at all, C/S denominators are noise and the
       benchmark would produce garbage (see src/metrics.py).

    2. SHUFFLED CONTROL: the same probe trained on shuffled labels must sit
       at chance on the held-out half. If it beats chance, evaluation is
       leaking (e.g. duplicated inputs straddling the split — another
       signature of the v1 gender data, where identical prefixes carried
       both labels).

run_benchmark refuses to launch unless every check passes.

The floors are registered per task on the Task classes (zc_gate_floor /
ze_gate_floor); they are provisional defaults until locked in
PREREGISTRATION_v2.md. The chance band for the shuffled control widens
automatically for small evaluation sets (binomial noise).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .probes import LinearProbe, ProbeTrainConfig, train_probe, probe_accuracy
from .repro import set_seed


@dataclass
class GateCheck:
    feature: str            # "zc" or "ze"
    accuracy: float         # held-out accuracy on true labels
    floor: float
    passed_floor: bool
    shuffled_accuracies: list[float]
    chance: float
    margin: float           # allowed |shuffled_acc - chance|
    passed_shuffle: bool

    @property
    def passed(self) -> bool:
        return self.passed_floor and self.passed_shuffle

    def as_dict(self) -> dict:
        return {
            "feature": self.feature,
            "accuracy": self.accuracy,
            "floor": self.floor,
            "passed_floor": self.passed_floor,
            "shuffled_accuracies": self.shuffled_accuracies,
            "chance": self.chance,
            "margin": self.margin,
            "passed_shuffle": self.passed_shuffle,
            "passed": self.passed,
        }


@dataclass
class GateResult:
    passed: bool
    n_train: int
    n_eval: int
    layer: int | None
    checks: list[GateCheck]

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "n_train": self.n_train,
            "n_eval": self.n_eval,
            "layer": self.layer,
            "checks": [c.as_dict() for c in self.checks],
        }

    def report(self) -> str:
        lines = []
        for c in self.checks:
            flag = "PASS" if c.passed else "FAIL"
            shuf = ", ".join(f"{a:.3f}" for a in c.shuffled_accuracies)
            lines.append(
                f"  [{flag}] {c.feature}: acc={c.accuracy:.3f} "
                f"(floor {c.floor:.2f}, floor {'ok' if c.passed_floor else 'FAILED'}); "
                f"shuffled=[{shuf}] (chance {c.chance:.2f} ± {c.margin:.3f}, "
                f"control {'ok' if c.passed_shuffle else 'FAILED'})"
            )
        verdict = "GATE PASSED" if self.passed else "GATE FAILED"
        return (f"{verdict} (layer={self.layer}, n_train={self.n_train}, "
                f"n_eval={self.n_eval})\n" + "\n".join(lines))


def _train_eval_linear(X_tr, y_tr, X_ev, y_ev, cfg, device, seed) -> float:
    set_seed(seed)
    probe = LinearProbe(X_tr.shape[1]).to(device)
    train_probe(probe, X_tr, y_tr, cfg, device)
    return probe_accuracy(probe, X_ev, y_ev, device)


def learnability_gate(
    X: torch.Tensor,
    zc: torch.Tensor,
    ze: torch.Tensor,
    device: torch.device,
    train_cfg: ProbeTrainConfig,
    zc_floor: float,
    ze_floor: float,
    chance: float = 0.5,
    n_shuffles: int = 3,
    base_margin: float = 0.05,
    layer: int | None = None,
    seed: int = 777,
) -> GateResult:
    """
    Run the learnability gate on one layer's representations.

    Splits (X, labels) in half deterministically, trains linear probes on
    the first half, and evaluates on the second. See module docstring for
    the two checks per feature.
    """
    n = X.shape[0]
    if n < 40:
        raise ValueError(f"learnability gate needs >= 40 examples, got {n}")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    half = n // 2
    tr_idx, ev_idx = perm[:half], perm[half:]
    n_eval = n - half

    # Standardize features INSIDE the gate (statistics from the train half
    # only). Linear decodability is scale-invariant, but the fixed-epoch
    # gate probe is not: layers with very small representation scale would
    # undertrain and produce false refusals. The benchmark itself still
    # trains on raw representations — this is gate-internal only.
    mu = X[tr_idx].mean(dim=0, keepdim=True)
    sd = X[tr_idx].std(dim=0, keepdim=True).clamp_min(1e-6)
    X = (X - mu) / sd

    # Binomial noise on the shuffled control's held-out accuracy: allow
    # 4 standard deviations around chance, never tighter than base_margin.
    margin = max(base_margin, 4.0 * (chance * (1 - chance) / n_eval) ** 0.5)

    checks: list[GateCheck] = []
    for feature, y, floor in (("zc", zc, zc_floor), ("ze", ze, ze_floor)):
        acc = _train_eval_linear(X[tr_idx], y[tr_idx], X[ev_idx], y[ev_idx],
                                 train_cfg, device, seed=seed + 1)
        shuffled_accs = []
        for i in range(n_shuffles):
            gs = torch.Generator().manual_seed(seed + 100 + i)
            y_shuf = y[torch.randperm(n, generator=gs)]
            shuffled_accs.append(
                _train_eval_linear(X[tr_idx], y_shuf[tr_idx],
                                   X[ev_idx], y_shuf[ev_idx],
                                   train_cfg, device, seed=seed + 200 + i)
            )
        checks.append(GateCheck(
            feature=feature,
            accuracy=acc,
            floor=floor,
            passed_floor=acc >= floor,
            shuffled_accuracies=shuffled_accs,
            chance=chance,
            margin=margin,
            passed_shuffle=all(abs(a - chance) <= margin
                               for a in shuffled_accs),
        ))

    return GateResult(
        passed=all(c.passed for c in checks),
        n_train=half,
        n_eval=n_eval,
        layer=layer,
        checks=checks,
    )
