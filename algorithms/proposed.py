"""Proposed FL Strategy - Dual-Heterogeneity Federated Learning (FedBN + SCAFFOLD).

This module implements the proposed strategy for the FedMedAI framework, simultaneously
tackling two orthogonal dimensions of heterogeneity in federated medical imaging:
1. Feature/Domain Shift (FedBN):
   Maintains local Batch Normalization layers per client to capture domain-specific
   scanner/staining distributions without cross-client parameter contamination.
2. Client Drift from Non-IID Label Skew (SCAFFOLD):
   Maintains global control variate c and local control variates c_i to estimate and
   subtract gradient drift during local training for non-BatchNorm trainable layers
   via Zero-Drift Masking.

Does NOT use FedNova step normalization (aggregates non-BN layers via sample-weighted FedAvg).

References:
    - Li et al., "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization",
      ICLR 2021. https://arxiv.org/abs/2102.07623
    - Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning",
      ICML 2020. https://arxiv.org/abs/1910.06378
"""

import time
from logging import INFO, WARNING
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch.nn as nn
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy

from algorithms.base import (
    FedStrategyBase,
    _format_device_details,
    _weighted_average_metrics,
)
from models.cnn import (
    get_bn_mask,
    get_buffer_mask,
    get_parameters,
    get_var_buffer_mask,
)
from monitoring.metrics import FLMetricsRecorder


def _fedbn_scaffold_aggregate(
    results: List[Tuple[ClientProxy, FitRes]],
    global_parameters: List[np.ndarray],
    server_control_variate: List[np.ndarray],
    bn_mask: List[bool],
    buffer_mask: List[bool],
    var_mask: List[bool],
    num_total_clients: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, float]]:
    """Aggregate model updates and control variates combining FedBN and SCAFFOLD.

    Args:
        results: List of (ClientProxy, FitRes) pairs from reporting clients.
        global_parameters: Current global model parameters (M ndarrays).
        server_control_variate: Current server control variates c (M ndarrays).
        bn_mask: Boolean mask indicating BatchNorm parameters/buffers.
        buffer_mask: Boolean mask indicating non-trainable buffers.
        var_mask: Boolean mask indicating variance tracking buffers (e.g. running_var).
        num_total_clients: Total clients N in the federation for scaling delta_c.

    Returns:
        Tuple of:
            - new_global_parameters (M ndarrays)
            - new_server_control_variate (M ndarrays)
            - stats dict (server_c_norm)
    """
    num_layers = len(global_parameters)

    # 0. Filter out corrupted or diverging client updates (NaN / Inf protection)
    valid_results = []
    for proxy, fit_res in results:
        arrays = parameters_to_ndarrays(fit_res.parameters)
        if any(np.isnan(a).any() or np.isinf(a).any() for a in arrays):
            cid = fit_res.metrics.get("client_id", "unknown")
            log(WARNING, "[Proposed] Discarding update from Client %s due to NaN/Inf values!", str(cid))
            continue
        valid_results.append((proxy, fit_res, arrays))

    if not valid_results:
        log(WARNING, "[Proposed] All client updates were invalid (NaN/Inf). Retaining global parameters.")
        return global_parameters, server_control_variate, {"server_c_norm": 0.0}

    total_examples = sum(fit_res.num_examples for _, fit_res, _ in valid_results)
    if total_examples == 0:
        return global_parameters, server_control_variate, {}

    client_models: List[List[np.ndarray]] = []
    client_delta_cs: List[List[np.ndarray]] = []
    client_weights: List[float] = []

    for _, fit_res, arrays in valid_results:
        if len(arrays) == 2 * num_layers:
            y_i = arrays[:num_layers]
            delta_c_i = arrays[num_layers:]
        else:
            y_i = arrays
            delta_c_i = [np.zeros_like(a, dtype=np.float64) for a in arrays]

        client_models.append(y_i)
        client_delta_cs.append(delta_c_i)

        p_i = fit_res.num_examples / total_examples
        client_weights.append(p_i)

    # 1. Model parameter aggregation (FedBN + standard sample-weighted averaging, no step scaling)
    new_params: List[np.ndarray] = []
    for l_idx in range(num_layers):
        g = global_parameters[l_idx]
        is_bn = l_idx < len(bn_mask) and bn_mask[l_idx]
        is_buf = l_idx < len(buffer_mask) and buffer_mask[l_idx]
        is_var = l_idx < len(var_mask) and var_mask[l_idx]

        if is_bn:
            # FedBN: Strictly preserve server global parameter (local to client)
            new_params.append(np.copy(g))
        elif is_buf:
            # Non-BN buffers: standard sample-weighted average clipped >= 0
            agg_buf = sum(
                p_i * np.asarray(c_model[l_idx], dtype=np.float64)
                for p_i, c_model in zip(client_weights, client_models)
            )
            if is_var:
                agg_buf = np.maximum(agg_buf, 0.0)
            new_params.append(agg_buf.astype(g.dtype))
        else:
            # Standard sample-weighted average for non-BN trainable parameters
            agg_layer = sum(
                p_i * np.asarray(c_model[l_idx], dtype=np.float64)
                for c_model, p_i in zip(client_models, client_weights)
            )
            if np.isnan(agg_layer).any() or np.isinf(agg_layer).any():
                new_layer = np.copy(g)
            else:
                new_layer = agg_layer.astype(g.dtype)
            new_params.append(new_layer)

    # 2. SCAFFOLD server control variate update (strictly for non-BN parameters)
    n_clients = max(num_total_clients, len(valid_results))
    new_server_c: List[np.ndarray] = []
    for l_idx in range(num_layers):
        is_bn = l_idx < len(bn_mask) and bn_mask[l_idx]
        old_c = np.asarray(server_control_variate[l_idx], dtype=np.float64)
        if is_bn:
            # BatchNorm layers remain zero-drift to protect local domain adaptation
            new_server_c.append(np.zeros_like(old_c, dtype=np.float64))
        else:
            sum_delta = sum(
                np.asarray(c_dc[l_idx], dtype=np.float64)
                for c_dc in client_delta_cs
                if not (np.isnan(c_dc[l_idx]).any() or np.isinf(c_dc[l_idx]).any())
            )
            c_l_new = old_c + (1.0 / n_clients) * sum_delta
            if np.isnan(c_l_new).any() or np.isinf(c_l_new).any():
                new_server_c.append(old_c)
            else:
                # Clip extreme control variate norm to prevent runaway drift
                c_norm_layer = float(np.linalg.norm(c_l_new))
                if c_norm_layer > 5.0:
                    c_l_new = c_l_new * (5.0 / c_norm_layer)
                new_server_c.append(c_l_new)

    c_norm = float(np.sqrt(sum(np.sum(arr ** 2) for arr in new_server_c)))

    stats = {
        "server_c_norm": float(c_norm),
    }

    return new_params, new_server_c, stats


class FedProposedStrategy(FedStrategyBase):
    """Proposed FL Strategy: FedBN + SCAFFOLD (Dual-Heterogeneity Federated Learning).

    Solves:
      1. Domain shift across client sites via local Batch Normalization (FedBN).
      2. Client drift across non-IID label distributions via control variates (SCAFFOLD)
         with Zero-Drift Masking on BatchNorm layers.

    Args:
        model: Required PyTorch model instance used for layer inspection and parameter shapes.
        num_total_clients: Total number of clients N in the federation for SCAFFOLD scaling.
        early_stop_patience: Number of rounds without improvement before stopping (0 to disable).
        early_stop_metric: Target metric to track ('accuracy' or 'loss').
        metrics_recorder: Optional FLMetricsRecorder instance.
        local_test: Whether local test partitions are enabled.
        **kwargs: Forwarded to flwr.server.strategy.FedAvg.
    """

    def __init__(
        self,
        *,
        model: Optional[nn.Module] = None,
        num_total_clients: Optional[int] = None,
        early_stop_patience: int = 10,
        early_stop_metric: str = "accuracy",
        metrics_recorder: Optional[FLMetricsRecorder] = None,
        local_test: bool = False,
        **kwargs,
    ):
        if model is None:
            raise ValueError(
                "FedProposedStrategy requires 'model' (nn.Module) to determine BatchNorm "
                "layers, buffer masks, and control variate shapes. Please pass model=<model_instance>."
            )

        super().__init__(
            early_stop_patience=early_stop_patience,
            early_stop_metric=early_stop_metric,
            metrics_recorder=metrics_recorder,
            model=model,
            local_test=local_test,
            strategy_name="Proposed",
            **kwargs,
        )

        self.num_total_clients = num_total_clients or self.min_fit_clients
        self.bn_mask = get_bn_mask(model)
        self.buffer_mask = get_buffer_mask(model)
        self.var_mask = get_var_buffer_mask(model)
        self.num_bn_layers = sum(self.bn_mask)
        self.num_non_bn_layers = len(self.bn_mask) - self.num_bn_layers

        initial_ndarrays = get_parameters(model)
        self.num_layers = len(initial_ndarrays)
        self.server_control_variate: List[np.ndarray] = [
            np.zeros_like(arr, dtype=np.float64) for arr in initial_ndarrays
        ]
        self._current_global_params: Optional[List[np.ndarray]] = None

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """Append server control variates c to global model parameters (2M arrays total)."""
        if self.should_stop:
            log(INFO, f"[{self.strategy_name}] EarlyStop: Training halted: configure_fit sampling 0 clients.")
            return []

        self._round_start = time.time()
        self.recorder.record_round_start(server_round)

        model_ndarrays = parameters_to_ndarrays(parameters)
        self._current_global_params = model_ndarrays
        self.num_layers = len(model_ndarrays)

        if len(self.server_control_variate) != self.num_layers:
            self.server_control_variate = [
                np.zeros_like(p, dtype=np.float64) for p in model_ndarrays
            ]

        combined_ndarrays = model_ndarrays + self.server_control_variate
        combined_parameters = ndarrays_to_parameters(combined_ndarrays)

        fit_ins_list = super(FedStrategyBase, self).configure_fit(
            server_round, combined_parameters, client_manager
        )

        for _, fit_ins in fit_ins_list:
            fit_ins.config["strategy"] = "proposed"
            fit_ins.config["is_fedbn"] = True
            fit_ins.config["is_scaffold"] = True
            fit_ins.config["server_round"] = server_round
            fit_ins.config["num_total_clients"] = self.num_total_clients

        return fit_ins_list

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Perform FedBN + SCAFFOLD aggregation without step normalization."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        if self._current_global_params is None:
            return None, {}

        aggregated_model, new_server_c, stats = _fedbn_scaffold_aggregate(
            results=results,
            global_parameters=self._current_global_params,
            server_control_variate=self.server_control_variate,
            bn_mask=self.bn_mask,
            buffer_mask=self.buffer_mask,
            var_mask=self.var_mask,
            num_total_clients=self.num_total_clients,
        )
        self.server_control_variate = new_server_c

        parameters_aggregated = ndarrays_to_parameters(aggregated_model)

        metrics_aggregated: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        metrics_aggregated["num_bn_layers_local"] = int(self.num_bn_layers)
        metrics_aggregated["num_non_bn_layers_aggregated"] = int(self.num_non_bn_layers)
        metrics_aggregated["server_c_norm"] = stats.get("server_c_norm", 0.0)

        metrics_aggregated = self._finalize_fit_aggregation(
            server_round=server_round,
            parameters_aggregated=parameters_aggregated,
            metrics_aggregated=metrics_aggregated,
            results=results,
            failures=failures,
        )

        log(
            INFO,
            "[Proposed: FedBN+SCAFFOLD] Round %d | %.1fs | clients=%d | c_norm=%.4f | local_bn=%d",
            server_round,
            metrics_aggregated["round_time_seconds"],
            len(results),
            stats.get("server_c_norm", 0.0),
            self.num_bn_layers,
        )

        return parameters_aggregated, metrics_aggregated

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        """Configure client evaluation round, informing clients to use local BN statistics."""
        if self.should_stop:
            return []
        evaluate_ins_list = super().configure_evaluate(
            server_round, parameters, client_manager
        )
        for _, evaluate_ins in evaluate_ins_list:
            evaluate_ins.config["strategy"] = "proposed"
            evaluate_ins.config["server_round"] = server_round
        return evaluate_ins_list

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Centralized server evaluation is bypassed in FedBN-based strategies."""
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Return a dict summarising the entire FL run."""
        summary = super().get_summary()
        summary["num_bn_layers_local"] = self.num_bn_layers
        summary["num_non_bn_layers_aggregated"] = self.num_non_bn_layers
        summary["server_c_norm"] = float(
            np.sqrt(sum(np.sum(arr ** 2) for arr in self.server_control_variate))
        )
        return summary

    def save_artifacts(
        self, save_dir: str, extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Export CSVs, comparison plots, and rich fl_results.json."""
        if extra_metadata is None:
            extra_metadata = {}
        extra_metadata["num_bn_layers_local"] = self.num_bn_layers
        extra_metadata["num_non_bn_layers_aggregated"] = self.num_non_bn_layers
        extra_metadata["server_c_norm"] = float(
            np.sqrt(sum(np.sum(arr ** 2) for arr in self.server_control_variate))
        )
        return super().save_artifacts(save_dir=save_dir, extra_metadata=extra_metadata)


# Canonical aliases
ProposedStrategy = FedProposedStrategy
FedBNScaffoldStrategy = FedProposedStrategy
ScaffoldFedBNStrategy = FedProposedStrategy

__all__ = [
    "FedProposedStrategy",
    "ProposedStrategy",
    "FedBNScaffoldStrategy",
    "ScaffoldFedBNStrategy",
    "_fedbn_scaffold_aggregate",
]
