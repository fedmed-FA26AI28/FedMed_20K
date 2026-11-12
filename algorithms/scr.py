"""Support-calibrated risk (SCR) for the team's Flower simulation.

The server reads *training* class counts from the simulated partition and
broadcasts only global class counts and the coverage normalizer. Each client
computes its local weighted CE and support-dependent classifier anchor. Model
aggregation remains sample-weighted FedAvg. This partition-metadata path has
no formal privacy guarantee and is intended for the simulated experiment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy

from algorithms.base import FedStrategyBase


@dataclass(frozen=True)
class SupportStatistics:
    counts: np.ndarray
    kappa: np.ndarray
    coverage: np.ndarray
    class_client_share: np.ndarray
    sample_weights: np.ndarray
    head_anchor: np.ndarray
    variance_factor: np.ndarray


def classifier_head(model: nn.Module) -> nn.Linear:
    """Return the final linear classifier for either research or team CNN."""
    head = getattr(model, "fc2", None)
    if head is None:
        head = getattr(model, "fc", None)
    if not isinstance(head, nn.Linear):
        raise TypeError("SCR requires a linear classifier named fc or fc2")
    return head


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


def client_support_parameters(
    local_counts: np.ndarray,
    global_counts: np.ndarray,
    coverage_sum: np.ndarray,
    total: int,
    clients: int,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Configure a client using its counts and two global aggregates only."""
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


def head_penalty(
    model: nn.Module,
    reference: tuple[torch.Tensor, torch.Tensor],
    anchor: torch.Tensor,
    coefficient: float,
) -> torch.Tensor:
    """Compute SCR classifier head drift penalty."""
    head = classifier_head(model)
    if coefficient == 0:
        return head.weight.new_zeros(())
    ref_w, ref_b = reference
    row_drift = (head.weight - ref_w).square().sum(dim=1)
    row_drift += (head.bias - ref_b).square()
    return 0.5 * coefficient * (anchor * row_drift).sum() / len(anchor)


def training_counts_from_partition(path: str | Path, num_classes: int,
                                   num_clients: int) -> np.ndarray:
    """Read and validate each client's training histogram, never test counts."""
    with open(path, encoding="utf-8") as file:
        partition = json.load(file)
    if int(partition["num_clients"]) != num_clients:
        raise ValueError("SCR partition client count differs from server client count")
    clients = partition["clients"]
    counts = np.zeros((num_clients, num_classes), dtype=np.int64)
    for client_id in range(num_clients):
        entry = clients[str(client_id)]
        histogram = entry.get("class_distribution")
        if histogram is None:
            raise ValueError("SCR requires training class_distribution in partition JSON")
        if any(int(label) < 0 or int(label) >= num_classes for label in histogram):
            raise ValueError("SCR partition contains a class outside model output range")
        counts[client_id] = [int(histogram.get(str(cls), 0)) for cls in range(num_classes)]
        if np.any(counts[client_id] < 0) or counts[client_id].sum() != int(entry["num_samples"]):
            raise ValueError(f"SCR client {client_id} training histogram is inconsistent")
        if "indices" in entry and len(entry["indices"]) != int(entry["num_samples"]):
            raise ValueError(f"SCR client {client_id} training index count is inconsistent")
    if counts.sum() != int(partition["total_samples"]):
        raise ValueError("SCR partition total_samples does not match training histograms")
    return counts


class SCRStrategy(FedStrategyBase):
    """FedAvg server with support-calibrated client risk configuration."""

    def __init__(
        self,
        *,
        model: nn.Module,
        partition_path: str | Path,
        num_total_clients: int,
        num_classes: Optional[int] = None,
        scr_gamma: float = 1.0,
        scr_head_lambda: float = 0.01,
        early_stop_patience: int = 0,
        early_stop_metric: str = "accuracy",
        metrics_recorder=None,
        local_test: bool = False,
        **kwargs,
    ):
        if model is None:
            raise ValueError("SCR requires the model to determine its classifier classes")
        if (num_total_clients < 2 or scr_head_lambda < 0
                or not np.isfinite(scr_head_lambda) or scr_gamma <= 0
                or not np.isfinite(scr_gamma)):
            raise ValueError("SCR requires at least two clients, finite positive gamma and nonnegative head lambda")
        detected_classes = classifier_head(model).out_features
        if num_classes is None:
            num_classes = detected_classes
        counts = training_counts_from_partition(partition_path, num_classes, num_total_clients)
        support = support_statistics(counts, scr_gamma)
        self.scr_gamma = float(scr_gamma)
        self.scr_head_lambda = float(scr_head_lambda)
        self.num_total_clients = num_total_clients
        self.partition_path = str(partition_path)
        self.global_counts = counts.sum(axis=0).astype(np.int64)
        self.coverage_sum = support.coverage.sum(axis=0)
        self.total_samples = int(counts.sum())
        self._scr_fit_config = {
            "scr_global_counts": json.dumps(self.global_counts.tolist()),
            "scr_coverage_sum": json.dumps(self.coverage_sum.tolist()),
            "scr_total_samples": self.total_samples,
            "scr_num_clients": num_total_clients,
            "scr_gamma": self.scr_gamma,
            "scr_head_lambda": self.scr_head_lambda,
        }
        super().__init__(
            model=model,
            early_stop_patience=early_stop_patience,
            early_stop_metric=early_stop_metric,
            metrics_recorder=metrics_recorder,
            local_test=local_test,
            strategy_name="SCR",
            **kwargs,
        )

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        instructions = super().configure_fit(server_round, parameters, client_manager)
        for _, fit_ins in instructions:
            fit_ins.config.update(self._scr_fit_config)
        return instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        metrics = self._finalize_fit_aggregation(
            server_round, parameters, metrics, results, failures)
        return parameters, metrics

    def get_summary(self) -> Dict[str, Any]:
        summary = super().get_summary()
        summary.update({"scr_gamma": self.scr_gamma,
                        "scr_head_lambda": self.scr_head_lambda,
                        "scr_partition_path": self.partition_path,
                        "scr_total_samples": self.total_samples})
        return summary


__all__ = [
    "SCRStrategy",
    "training_counts_from_partition",
    "classifier_head",
    "support_statistics",
    "client_support_parameters",
    "head_penalty",
    "SupportStatistics",
]
