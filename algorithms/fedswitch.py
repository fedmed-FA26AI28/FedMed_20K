"""FedSwitch Strategy: Balanced Aggregation via Private Histogram Estimation.

Reference:
    "FedSwitch: From Standard to Balanced Aggregation via Private Histogram Estimation in
     Federated Medical Image Classification"
    Paolo Cacace, Riccardo Taiello, Andrea Protani, Marc Molina Van den Bosch,
    Diogo Reis Santos, Pierpaolo Brutti, Luigi Serio (MICCAI 2026 / CERN).

FedSwitch mitigates class imbalance in federated medical imaging:
1. Warm-up Phase (rounds 1 <= r <= R*): Aggregates client updates with standard FedAvg weights.
2. Switching Round (R*): Estimates the global label histogram under client-level epsilon-DP.
   The optimal switching round is calculated via closed-form coverage:
       R* = ceil( ln(1 - rho) / ln(1 - m/N) )
3. Balanced Phase (rounds r > R*): Replaces dataset-size weights with inverse-frequency weights
   derived from the private histogram estimate, upweighting clients with rare pathologies.
"""

from __future__ import annotations

import json
from logging import INFO, WARNING
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from algorithms.base import FedStrategyBase


def _sample_discrete_laplace(scale: float, size: int = 1) -> np.ndarray:
    """Sample from Discrete Laplace distribution DLap(scale) using geometric difference."""
    if scale <= 1e-12:
        return np.zeros(size, dtype=np.int64)
    p = 1.0 - np.exp(-1.0 / scale)
    g1 = np.random.geometric(p, size=size) - 1
    g2 = np.random.geometric(p, size=size) - 1
    return (g1 - g2).astype(np.int64)


def compute_switching_round(
    total_clients: int,
    sampled_per_round: int,
    coverage: float = 0.9,
    total_rounds: Optional[int] = None,
) -> int:
    """Calculate the earliest communication round R* achieving target client coverage rho.

    Formula: R* = ceil( log(1 - rho) / log(1 - m/N) )
    When m >= N (all clients participate every round), coverage is immediately 100%.
    In accordance with the FedSwitch paper's warm-up requirement (~20-25% of total budget),
    we allocate a 25% warm-up phase (e.g. Round 20 for 80 rounds, or 20 by default).
    """
    N = max(int(total_clients), 1)
    m = max(int(sampled_per_round), 1)
    if m >= N:
        if total_rounds is not None and int(total_rounds) > 0:
            return max(1, int(0.25 * int(total_rounds)))
        return 20  # Paper-aligned default warm-up (e.g. R*=21 in MICCAI 2024 paper)
    rho = min(max(float(coverage), 0.01), 0.999)
    ratio = 1.0 - (float(m) / float(N))
    r_star = int(np.ceil(np.log(1.0 - rho) / np.log(ratio)))
    return max(r_star, 1)


# Convenient public aliases
sample_discrete_laplace = _sample_discrete_laplace
compute_optimal_switch_round = compute_switching_round


class FedSwitchStrategy(FedStrategyBase):
    """FedSwitch: Standard to Balanced Aggregation via Private Histogram Estimation.

    Args:
        switch_round: Explicit switching round R*. If None, auto-calculated via coverage formula.
        switch_coverage: Target client coverage fraction rho (default: 0.9).
        switch_epsilon: Privacy budget epsilon for Discrete Laplace noise (default: 1.0).
        switch_c_bound: Contribution bound C (sensitivity). If None, inferred or defaults to 50.
        partition_path: Optional path to partition JSON file with ground-truth distributions.
        num_classes: Optional total number of classes.
        num_total_clients: Total clients N in the federation (default: 12).
        weighting_mode: 'bounded_sum' (min(1, sum(c/y)), MICCAI 2024 paper default) or 'stratified' (mean(c/y)).
        model: Global model instance for inspection.
        **kwargs: Forwarded to FedStrategyBase.
    """

    def __init__(
        self,
        *,
        switch_round: Optional[int] = None,
        switch_coverage: float = 0.9,
        switch_epsilon: float = 1.0,
        switch_c_bound: Optional[int] = None,
        partition_path: Optional[Union[str, Path]] = None,
        num_classes: Optional[int] = None,
        num_total_clients: int = 12,
        weighting_mode: str = "bounded_sum",
        total_rounds: Optional[int] = None,
        model: Optional[nn.Module] = None,
        strategy_name: str = "FedSwitch",
        **kwargs,
    ):
        super().__init__(
            model=model,
            strategy_name=strategy_name,
            **kwargs,
        )
        self.num_total_clients = int(num_total_clients)
        self.switch_coverage = float(switch_coverage)
        self.switch_epsilon = max(float(switch_epsilon), 1e-4)
        self.switch_c_bound = int(switch_c_bound) if switch_c_bound is not None else 50
        self.partition_path = Path(partition_path) if partition_path else None
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.total_rounds = int(total_rounds) if total_rounds is not None else None
        self.weighting_mode = str(weighting_mode).lower().strip()
        if self.weighting_mode not in ("bounded_sum", "stratified"):
            raise ValueError(f"Unknown weighting_mode: '{self.weighting_mode}'. Must be 'bounded_sum' or 'stratified'.")

        # Determine sampling rate m from kwargs or default
        min_fit = kwargs.get("min_fit_clients", kwargs.get("min_clients", self.num_total_clients))
        self.sampled_per_round = max(int(min_fit), 1)

        # Determine switching round R*
        if switch_round is not None and int(switch_round) >= 1:
            self.switch_round = int(switch_round)
            log(INFO, "[FedSwitch] Using explicit switch_round: %d", self.switch_round)
        else:
            self.switch_round = compute_switching_round(
                total_clients=self.num_total_clients,
                sampled_per_round=self.sampled_per_round,
                coverage=self.switch_coverage,
                total_rounds=self.total_rounds,
            )
            log(
                INFO,
                "[FedSwitch] Auto-computed switch_round: R*=%d (N=%d, m=%d, coverage=%.2f, mode=%s)",
                self.switch_round,
                self.num_total_clients,
                self.sampled_per_round,
                self.switch_coverage,
                self.weighting_mode,
            )
            log(
                INFO,
                "[FedSwitch] Auto-computed switch_round: R*=%d (N=%d, m=%d, coverage=%.2f)",
                self.switch_round,
                self.num_total_clients,
                self.sampled_per_round,
                self.switch_coverage,
            )

        # Client counts and estimated histogram
        self.client_class_counts: Dict[int, np.ndarray] = {}
        self.estimated_histogram: Optional[np.ndarray] = None
        self._load_partition_distributions()

    def _load_partition_distributions(self) -> None:
        """Pre-populate partition distributions if a partition JSON path is supplied."""
        if not self.partition_path or not self.partition_path.exists():
            return

        try:
            with open(self.partition_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            clients = data.get("clients", {})
            if not clients:
                return

            if self.num_classes is None:
                max_cls = 0
                for entry in clients.values():
                    dist = entry.get("class_distribution", {})
                    for c_str in dist.keys():
                        max_cls = max(max_cls, int(c_str))
                self.num_classes = max_cls + 1

            for cid_str, entry in clients.items():
                cid = int(cid_str)
                dist = entry.get("class_distribution", {})
                counts = np.zeros(self.num_classes, dtype=np.float32)
                for c_str, count in dist.items():
                    c = int(c_str)
                    if c < self.num_classes:
                        counts[c] = float(count)
                self.client_class_counts[cid] = counts

            log(
                INFO,
                "[FedSwitch] Loaded ground truth distributions for %d clients (num_classes=%d) from partition",
                len(self.client_class_counts),
                self.num_classes,
            )
        except Exception as exc:
            log(WARNING, "[FedSwitch] Could not parse partition JSON: %s", exc)

    def _estimate_dp_histogram(self) -> np.ndarray:
        """Estimate the global label histogram under client-level epsilon-DP with Discrete Laplace noise."""
        if self.num_classes is None:
            self.num_classes = 10

        total_counts = np.zeros(self.num_classes, dtype=np.float64)
        for counts in self.client_class_counts.values():
            if len(counts) == self.num_classes:
                total_counts += counts

        # Compute noise scale C / epsilon
        sensitivity = float(self.switch_c_bound)
        scale = sensitivity / self.switch_epsilon
        dp_noise = _sample_discrete_laplace(scale=scale, size=self.num_classes)

        noisy_histogram = np.maximum(total_counts + dp_noise, 1.0)
        log(
            INFO,
            "[FedSwitch] Reconstructed DP Global Histogram (eps=%.2f, C=%d): %s (noise=%s)",
            self.switch_epsilon,
            self.switch_c_bound,
            [round(x, 1) for x in noisy_histogram.tolist()],
            dp_noise.tolist(),
        )
        return noisy_histogram

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure client fit instructions, broadcasting strategy name and phase."""
        client_instructions = super().configure_fit(server_round, parameters, client_manager)

        phase = "warmup_fedavg" if server_round <= self.switch_round else "balanced_aggregation"
        for _, fit_ins in client_instructions:
            fit_ins.config["strategy"] = "fedswitch"
            fit_ins.config["fedswitch_phase"] = phase
            fit_ins.config["fedswitch_switch_round"] = self.switch_round

        return client_instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates using FedAvg (round <= R*) or Balanced Aggregation (round > R*)."""
        if not results:
            return None, {}

        # Harvest client class distribution reports if sent via metrics
        for _, fit_res in results:
            cid = fit_res.metrics.get("client_id")
            if cid is not None:
                cid_int = int(cid)
                if "class_counts_json" in fit_res.metrics:
                    try:
                        c_dict = json.loads(str(fit_res.metrics["class_counts_json"]))
                        if self.num_classes is None:
                            self.num_classes = max(int(k) for k in c_dict.keys()) + 1
                        arr = np.zeros(self.num_classes, dtype=np.float32)
                        for k, v in c_dict.items():
                            arr[int(k)] = float(v)
                        self.client_class_counts[cid_int] = arr
                    except Exception:
                        pass

        # Check if we reached switching round R* and need to reconstruct DP histogram
        if server_round >= self.switch_round and self.estimated_histogram is None:
            self.estimated_histogram = self._estimate_dp_histogram()
            log(
                INFO,
                "[FedSwitch] Triggered switch at Round %d -> Switched from FedAvg to Balanced Aggregation!",
                server_round,
            )

        # Phase 1: Warm-up with standard FedAvg weights
        if server_round <= self.switch_round or self.estimated_histogram is None:
            log(INFO, "[FedSwitch] Round %d: Warm-up phase (standard FedAvg weighting)", server_round)
            return super().aggregate_fit(server_round, results, failures)

        # Phase 2: Balanced Aggregation using inverse-frequency weights
        log(INFO, "[FedSwitch] Round %d: Balanced phase (distribution-aware weighting)", server_round)

        unnormalized_weights: List[float] = []
        parsed_results: List[Tuple[np.ndarray, FitRes]] = []
        num_classes = self.num_classes or len(self.estimated_histogram)
        denom_histogram = np.maximum(self.estimated_histogram, 1.0)

        for client_proxy, fit_res in results:
            cid = fit_res.metrics.get("client_id")
            cid_int = int(cid) if cid is not None else None

            # Retrieve client label counts
            if cid_int is not None and cid_int in self.client_class_counts:
                counts = self.client_class_counts[cid_int]
            else:
                counts = np.ones(num_classes, dtype=np.float32) * (fit_res.num_examples / float(num_classes))

            # Compute distribution-aware score mu_i
            if self.weighting_mode == "bounded_sum":
                mu_i = float(np.minimum(1.0, np.sum(counts / denom_histogram)))
            else:
                # Stratified mean ratio: (1 / K) * sum_c (c_{i,c} / hat_y_c)
                mu_i = float(np.mean(counts / denom_histogram))

            unnormalized_weights.append(max(mu_i, 1e-8))
            param_ndarrays = parameters_to_ndarrays(fit_res.parameters)
            parsed_results.append((param_ndarrays, fit_res))

        # Normalize weights
        total_weight = sum(unnormalized_weights)
        norm_weights = [w / total_weight for w in unnormalized_weights]

        # Parameter aggregation
        first_ndarrays, _ = parsed_results[0]
        aggregated_ndarrays = [np.zeros_like(layer, dtype=np.float32) for layer in first_ndarrays]

        for (client_ndarrays, _), weight in zip(parsed_results, norm_weights):
            for agg_layer, client_layer in zip(aggregated_ndarrays, client_ndarrays):
                agg_layer += client_layer.astype(np.float32) * weight

        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)
        self.latest_parameters = aggregated_parameters

        # Aggregate metrics
        metrics_dict: Dict[str, Scalar] = {
            "fedswitch_phase": "balanced",
            "switch_round": self.switch_round,
            "fedswitch_weight_min": float(min(norm_weights)),
            "fedswitch_weight_max": float(max(norm_weights)),
        }
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            aggregated_fit_metrics = self.fit_metrics_aggregation_fn(fit_metrics)
            metrics_dict.update(aggregated_fit_metrics)

        if aggregated_parameters is not None or metrics_dict:
            metrics_dict = self._finalize_fit_aggregation(
                server_round=server_round,
                parameters_aggregated=aggregated_parameters,
                metrics_aggregated=metrics_dict,
                results=results,
                failures=failures,
            )

        return aggregated_parameters, metrics_dict
