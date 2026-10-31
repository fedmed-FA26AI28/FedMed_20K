"""Support-calibrated risk (SCR) for the team's Flower simulation.

The server reads *training* class counts from the simulated partition and
broadcasts only global class counts and the coverage normalizer. Each client
computes its local weighted CE and support-dependent classifier anchor. Model
aggregation remains sample-weighted FedAvg. This partition-metadata path has
no formal privacy guarantee and is intended for the simulated experiment.
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
from algorithms.research_methods.models import classifier_head
from algorithms.research_methods.support import support_statistics


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
        num_classes = classifier_head(model).out_features
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


__all__ = ["SCRStrategy", "training_counts_from_partition"]
