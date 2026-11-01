"""FedSLD Strategy: Federated Learning with Shared Label Distribution.

Reference:
    "FedSLD: Federated Learning with Shared Label Distribution for Medical Image Classification"
    Jun Luo & Shandong Wu (IEEE ISBI 2022).

FedSLD addresses label distribution skew by sharing only marginal label frequencies (class counts),
deriving a global federation prior P~(y), and dynamically scaling each sample's local mini-batch
cross-entropy gradient based on the ratio p_b(y) / P~(y).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch.nn as nn
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy

from algorithms.base import FedStrategyBase


def load_prior_from_partition(
    partition_path: str | Path,
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """Read client training histograms from partition JSON and compute global prior distribution P~(y)."""
    p_path = Path(partition_path)
    if not p_path.exists():
        raise FileNotFoundError(f"Partition file not found: {partition_path}")

    with open(p_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    clients = data.get("clients", {})
    if not clients:
        raise ValueError(f"No client partition entries found in {partition_path}")

    # Determine num_classes if not provided
    if num_classes is None:
        max_cls = 0
        for entry in clients.values():
            dist = entry.get("class_distribution", {})
            for c_str in dist.keys():
                max_cls = max(max_cls, int(c_str))
        num_classes = max_cls + 1

    total_counts = np.zeros(num_classes, dtype=np.float64)
    for entry in clients.values():
        dist = entry.get("class_distribution", {})
        for c_str, count in dist.items():
            c = int(c_str)
            if c < num_classes:
                total_counts[c] += float(count)

    total_samples = total_counts.sum()
    if total_samples <= 0:
        return np.ones(num_classes, dtype=np.float64) / num_classes

    return total_counts / total_samples


class FedSLDStrategy(FedStrategyBase):
    """Federated Learning with Shared Label Distribution (FedSLD) Server Strategy."""

    def __init__(
        self,
        *,
        model: Optional[nn.Module] = None,
        partition_path: Optional[str | Path] = None,
        prior_distribution: Optional[Union[List[float], np.ndarray]] = None,
        fedsld_epsilon: float = 1e-6,
        fedsld_reduction: str = "mean",
        num_classes: Optional[int] = None,
        early_stop_patience: int = 0,
        early_stop_metric: str = "accuracy",
        metrics_recorder=None,
        local_test: bool = False,
        **kwargs,
    ):
        """Initialize FedSLDStrategy.

        Args:
            model: PyTorch model instance (optional, used to infer num_classes if classifier head exists).
            partition_path: Path to the training partition JSON to compute global class distribution P~(y).
            prior_distribution: Explicit global prior P~(y) if known beforehand.
            fedsld_epsilon: Numerical safeguard epsilon to prevent division by zero (default: 1e-6).
            fedsld_reduction: Batch loss reduction mode ('mean' or 'sum', default: 'mean').
            num_classes: Number of classification targets.
            early_stop_patience: Rounds of stagnation before halting early.
            early_stop_metric: Metric to monitor for early stopping ('accuracy' or 'loss').
            metrics_recorder: FLMetricsRecorder instance.
            local_test: Whether local test evaluation is active.
        """
        # Resolve number of classes from model if possible
        if num_classes is None and model is not None:
            for m in reversed(list(model.modules())):
                if isinstance(m, nn.Linear):
                    num_classes = m.out_features
                    break

        # Compute or assign global federation prior P~(y)
        if prior_distribution is not None:
            prior = np.asarray(prior_distribution, dtype=np.float64)
            prior = prior / prior.sum()
        elif partition_path is not None and Path(partition_path).exists():
            prior = load_prior_from_partition(partition_path, num_classes=num_classes)
        elif num_classes is not None:
            prior = np.ones(num_classes, dtype=np.float64) / num_classes
        else:
            prior = np.ones(8, dtype=np.float64) / 8.0  # MedMNIST standard fallback

        self.global_prior = prior
        self.partition_path = str(partition_path) if partition_path else None
        self.fedsld_epsilon = float(fedsld_epsilon)
        self.fedsld_reduction = str(fedsld_reduction).lower()

        self._fedsld_fit_config = {
            "fedsld_prior": json.dumps(self.global_prior.tolist()),
            "fedsld_epsilon": self.fedsld_epsilon,
            "fedsld_reduction": self.fedsld_reduction,
        }

        super().__init__(
            model=model,
            early_stop_patience=early_stop_patience,
            early_stop_metric=early_stop_metric,
            metrics_recorder=metrics_recorder,
            local_test=local_test,
            strategy_name="FedSLD",
            **kwargs,
        )

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """Broadcast global prior P~(y) and FedSLD hyperparameters to clients in fit config."""
        instructions = super().configure_fit(server_round, parameters, client_manager)
        for _, fit_ins in instructions:
            fit_ins.config.update(self._fedsld_fit_config)
        return instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Perform FedAvg sample-weighted aggregation and log metrics."""
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        metrics = self._finalize_fit_aggregation(
            server_round, parameters, metrics, results, failures
        )
        return parameters, metrics

    def get_summary(self) -> Dict[str, Any]:
        """Return strategy metadata and hyperparameters for reporting."""
        summary = super().get_summary()
        summary.update({
            "fedsld_epsilon": self.fedsld_epsilon,
            "fedsld_reduction": self.fedsld_reduction,
            "fedsld_partition_path": self.partition_path,
            "fedsld_prior": self.global_prior.tolist(),
        })
        return summary


__all__ = ["FedSLDStrategy", "load_prior_from_partition"]
