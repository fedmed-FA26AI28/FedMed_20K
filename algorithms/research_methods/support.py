"""Training-count statistics for support-calibrated risk and its controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from algorithms.research_methods.models import classifier_head


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
    """Build fixed weights from training counts only.

    At gamma=inf the data term becomes pooled class-balanced CE. A class absent
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
    means = np.asarray(class_means, dtype=np.float64)
    if means.shape != stats.counts.shape:
        raise ValueError("class_means shape does not match counts")
    supported = stats.counts.sum(axis=0) > 0
    return float(np.sum(stats.class_client_share * means) / supported.sum())


def client_support_parameters(local_counts: np.ndarray, global_counts: np.ndarray,
                              coverage_sum: np.ndarray, total: int, clients: int,
                              gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Configure a client using its counts and two global aggregates only.

    This helper does not implement secure aggregation or a privacy guarantee.
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
    """FedSwitch aggregation comparator with an exact, non-private histogram."""
    n = np.asarray(counts, dtype=np.float64)
    if n.ndim != 2 or np.any(n < 0) or not np.all(np.isfinite(n)):
        raise ValueError("counts must be a finite nonnegative K x C matrix")
    global_counts = n.sum(axis=0)
    scores = np.minimum(1.0, (n / np.maximum(1.0, global_counts)).sum(axis=1))
    if scores.sum() <= 0:
        raise ValueError("at least one class must be observed")
    return scores / scores.sum()


def head_penalty(model: nn.Module, reference: tuple[torch.Tensor, torch.Tensor],
                 anchor: torch.Tensor, coefficient: float) -> torch.Tensor:
    head = classifier_head(model)
    if coefficient == 0:
        return head.weight.new_zeros(())
    ref_w, ref_b = reference
    row_drift = (head.weight - ref_w).square().sum(dim=1)
    row_drift += (head.bias - ref_b).square()
    return 0.5 * coefficient * (anchor * row_drift).sum() / len(anchor)
