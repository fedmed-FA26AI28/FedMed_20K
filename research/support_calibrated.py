"""Support-calibrated class-client risk and matched research losses.

The data part of the proposed objective has an exact FedAvg gradient identity
when every client takes one gradient step from the same model. Multi-step local
training and the classifier anchor are additional optimization choices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


METHODS = (
    "fedavg", "fedsld_literal", "fedlc", "fedvls", "fedswitch_oracle", "coverage_iw",
    "pooled_uniform_head", "coverage_head_only", "support_uniform_head",
    "support_calibrated",
)


class TinyCNNGN(nn.Module):
    """Shared 2D MedMNIST backbone with no client-specific running statistics."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.MaxPool2d(2), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.fc(self.encoder(images))


class TinyCNNBN(TinyCNNGN):
    """The same CNN with BatchNorm in place of GroupNorm.

    BN affine parameters and running statistics are shared through FedAvg.
    This is global BatchNorm, not FedBN's client-local normalization.
    """

    def __init__(self, num_classes: int):
        super().__init__(num_classes)
        self.encoder[1] = nn.BatchNorm2d(32)
        self.encoder[4] = nn.BatchNorm2d(64)


def make_model(num_classes: int, norm: str = "gn") -> TinyCNNGN:
    if norm == "gn":
        return TinyCNNGN(num_classes)
    if norm == "bn":
        return TinyCNNBN(num_classes)
    raise ValueError(f"unknown normalization: {norm}")


@dataclass(frozen=True)
class SupportStatistics:
    counts: np.ndarray
    kappa: np.ndarray
    coverage: np.ndarray
    class_client_share: np.ndarray
    sample_weights: np.ndarray
    head_anchor: np.ndarray
    variance_factor: np.ndarray


def support_statistics(counts: np.ndarray, gamma: float) -> SupportStatistics:
    """Build fixed weights from *training* counts only.

    For gamma=inf, the data loss is pooled class-balanced CE. A class absent
    federation-wide has no data term and is excluded from the uniform target.
    """
    n = np.asarray(counts, dtype=np.float64)
    if n.ndim != 2 or not np.all(np.isfinite(n)) or np.any(n < 0):
        raise ValueError("counts must be a finite nonnegative K x C matrix")
    if not np.all(n == np.floor(n)) or np.any(n.sum(axis=1) == 0):
        raise ValueError("counts must be integer-valued and each client nonempty")
    if gamma <= 0 or np.isnan(gamma):
        raise ValueError("gamma must be positive")
    k, c = n.shape
    total_per_class = n.sum(axis=0)
    supported = total_per_class > 0
    if not np.any(supported):
        raise ValueError("at least one class must be observed")
    total = n.sum()
    if np.isinf(gamma):
        coverage = np.divide(n, total_per_class, out=np.zeros_like(n), where=supported)
        share = coverage.copy()
        weights = np.broadcast_to(
            np.divide(total, supported.sum() * total_per_class,
                      out=np.zeros(c), where=supported), n.shape
        ).copy()
        weights[n == 0] = 0
        anchor = np.ones_like(n)
        kappa = np.full(c, np.inf)
    else:
        kappa = gamma * total_per_class / k
        coverage = np.divide(n, n + kappa, out=np.zeros_like(n),
                             where=(n + kappa) > 0)
        z = coverage.sum(axis=0)
        share = np.divide(coverage, z, out=np.zeros_like(n), where=z > 0)
        weights = np.divide(total, supported.sum() * z * (n + kappa),
                            out=np.zeros_like(n), where=(n > 0) & (z > 0))
        anchor = 1.0 - coverage
    variance = np.divide(share ** 2, n, out=np.zeros_like(n), where=n > 0).sum(axis=0)
    return SupportStatistics(n, kappa, coverage, share, weights, anchor, variance)


def empirical_target_risk(class_means: np.ndarray, stats: SupportStatistics) -> float:
    """Risk represented by the supported class-client groups."""
    means = np.asarray(class_means, dtype=np.float64)
    if means.shape != stats.counts.shape:
        raise ValueError("class_means shape does not match counts")
    supported = stats.counts.sum(axis=0) > 0
    return float(np.sum(stats.class_client_share * means) / supported.sum())


def client_support_parameters(local_counts: np.ndarray, global_counts: np.ndarray,
                              coverage_sum: np.ndarray, total: int, clients: int,
                              gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Configure one client using its own counts and two global aggregates only.

    The server can aggregate global_counts and coverage_sum in two passes. This
    helper does not implement secure aggregation or a privacy guarantee.
    """
    local = np.asarray(local_counts, dtype=np.float64)
    global_n = np.asarray(global_counts, dtype=np.float64)
    z = np.asarray(coverage_sum, dtype=np.float64)
    if (local.ndim != 1 or local.shape != global_n.shape or z.shape != local.shape
            or not np.all(np.isfinite(local)) or not np.all(np.isfinite(global_n))
            or not np.all(np.isfinite(z)) or np.any(local < 0) or np.any(local > global_n)
            or total <= 0 or clients <= 0 or gamma <= 0 or np.isnan(gamma)):
        raise ValueError("invalid local or aggregate class counts")
    supported = global_n > 0
    if not np.any(supported) or np.any(z[supported] <= 0):
        raise ValueError("every globally observed class needs positive coverage")
    if np.isinf(gamma):
        weight = np.divide(total, supported.sum() * global_n,
                           out=np.zeros_like(local), where=(local > 0))
        return weight, np.ones_like(local)
    kappa = gamma * global_n / clients
    coverage = np.divide(local, local + kappa, out=np.zeros_like(local),
                         where=(local + kappa) > 0)
    weight = np.divide(total, supported.sum() * z * (local + kappa),
                       out=np.zeros_like(local), where=(local > 0) & (z > 0))
    return weight, 1.0 - coverage


def fedswitch_oracle_weights(counts: np.ndarray) -> np.ndarray:
    """FedSwitch aggregation rule with an exact global histogram, no privacy claim.

    In our full-participation simulation the histogram is available after round
    one, so these weights apply from round two. The paper uses an estimated,
    client-level DP histogram instead of this exact oracle.
    """
    n = np.asarray(counts, dtype=np.float64)
    if n.ndim != 2 or np.any(n < 0) or not np.all(np.isfinite(n)):
        raise ValueError("counts must be a finite nonnegative K x C matrix")
    global_counts = n.sum(axis=0)
    scores = np.minimum(1.0, (n / np.maximum(1.0, global_counts)).sum(axis=1))
    if scores.sum() <= 0:
        raise ValueError("at least one class must be observed")
    return scores / scores.sum()


def head_penalty(model: TinyCNNGN, reference: tuple[torch.Tensor, torch.Tensor],
                 anchor: torch.Tensor, coefficient: float) -> torch.Tensor:
    if coefficient == 0:
        return model.fc.weight.new_zeros(())
    ref_w, ref_b = reference
    row_drift = (model.fc.weight - ref_w).square().sum(dim=1)
    row_drift += (model.fc.bias - ref_b).square()
    return 0.5 * coefficient * (anchor * row_drift).sum() / len(anchor)


def local_objective(
    method: str, model: TinyCNNGN, images: torch.Tensor,
    labels: torch.Tensor, counts: torch.Tensor, global_counts: torch.Tensor,
    sample_weights: torch.Tensor, anchor: torch.Tensor,
    reference: tuple[torch.Tensor, torch.Tensor], head_lambda: float = 0.0,
    fedlc_tau: float = 0.5, teacher: TinyCNNGN | None = None,
    fedvls_lambda: float = 0.1,
) -> torch.Tensor:
    """One local minibatch loss. All evaluation uses unmodified model logits."""
    logits = model(images)
    ce = F.cross_entropy(logits, labels, reduction="none")
    if method in ("fedavg", "fedswitch_oracle"):
        return ce.mean()
    if method == "coverage_head_only":
        return ce.mean() + head_penalty(model, reference, anchor, head_lambda)
    if method == "fedsld_literal":
        batch_prior = torch.bincount(labels, minlength=logits.shape[1]).to(logits) / len(labels)
        global_prior = global_counts / global_counts.sum()
        return (batch_prior[labels] / global_prior[labels] * ce).sum()
    if method == "coverage_iw":
        batch_counts = torch.bincount(labels, minlength=logits.shape[1]).to(logits)
        batch_prior = (batch_counts + 1) / (len(labels) + logits.shape[1])
        global_prior = (global_counts + 1) / (global_counts.sum() + logits.shape[1])
        ratio = global_prior[labels] / batch_prior[labels]
        inverse_loss = (ratio * ce).sum() / ratio.sum()
        # coverage_iw.md uses mu/2 * sum_c a_kc ||head_c-global_c||^2.
        return inverse_loss + head_penalty(
            model, reference, anchor, head_lambda * len(anchor))
    if method == "fedlc":
        offset = fedlc_tau * counts.clamp_min(1).pow(-0.25)
        if fedlc_tau > 0:
            offset = offset.masked_fill(counts == 0, float("inf"))
        return F.cross_entropy(logits - offset, labels)
    if method == "fedvls":
        # Adapted mathematical losses from the authors' released clientvls.py.
        # Model, optimizer, data pipeline and FedAvg protocol remain matched here.
        observed = counts > 0
        prior = counts / counts.sum()
        log_prior = (prior + 1e-9).log()
        loss = F.cross_entropy(logits + log_prior, labels)
        adjusted = logits - log_prior + logits.new_tensor(1 / logits.shape[1]).log()
        nonlabel = labels[:, None] != torch.arange(logits.shape[1], device=logits.device)
        spread = torch.where(nonlabel, adjusted, torch.zeros_like(adjusted))
        suppress = (prior * (torch.logsumexp(spread, dim=0) -
                             logits.new_tensor(len(labels)).log())).sum()
        loss = loss + 0.005 * suppress
        vacant = torch.where(~observed)[0]
        if teacher is not None and len(vacant) > 1:
            with torch.no_grad():
                teacher_prob = F.softmax(teacher(images)[:, vacant], dim=1)
            # Official client code uses KL(global vacant distribution || local).
            loss = loss + fedvls_lambda * F.kl_div(
                F.log_softmax(logits[:, vacant], dim=1), teacher_prob,
                reduction="batchmean",
            )
        return loss
    if method not in ("pooled_uniform_head", "support_uniform_head", "support_calibrated"):
        raise ValueError(f"unknown research method: {method}")
    loss = (sample_weights[labels] * ce).mean()
    return loss + head_penalty(model, reference, anchor, head_lambda)
