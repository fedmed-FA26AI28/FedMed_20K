"""FedNova Strategy - Normalized Averaging for heterogeneous client compute.

FedNova adjusts the aggregation weights by the number of local steps taken
by each client, eliminating objective inconsistency caused by heterogeneous
local updates (e.g., fast clients taking more steps than slow clients).
"""

from logging import INFO, WARNING
from typing import Dict, List, Optional, Tuple, Union

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
from models.cnn import get_buffer_mask, get_var_buffer_mask
from monitoring.metrics import FLMetricsRecorder


def _fednova_aggregate(
    results: List[Tuple[ClientProxy, FitRes]],
    global_parameters: List[np.ndarray],
    buffer_mask: Optional[List[bool]] = None,
    var_mask: Optional[List[bool]] = None,
) -> List[np.ndarray]:
    """Aggregate model updates using FedNova normalized averaging.

    Each client i computes:
        delta_i = w_i - w_global
        d_i = delta_i / tau_i  (normalized direction, tau_i = local_steps)

    Effective number of steps:
        tau_eff = sum_i(p_i * tau_i)  where p_i = n_i / n_total

    Aggregated update:
        For trainable parameters:
            delta_nova = tau_eff * sum_i(p_i * d_i)
            w_new = w_global + delta_nova
        For non-trainable buffers (e.g. BatchNorm running_mean, running_var):
            Standard sample-weighted averaging (FedAvg) is used because buffers
            represent empirical activation statistics rather than optimization
            variables updated by SGD. Variance buffers are strictly clipped >= 0
            to prevent numerical instability and NaN losses during evaluation.
    """
    total_examples = sum(fit_res.num_examples for _, fit_res in results)
    if total_examples == 0:
        return global_parameters

    client_deltas = []
    client_weights = []
    client_taus = []
    client_params_list = []

    for _, fit_res in results:
        client_params = parameters_to_ndarrays(fit_res.parameters)
        client_params_list.append(client_params)
        delta = [
            np.asarray(c, dtype=np.float64) - np.asarray(g, dtype=np.float64)
            for c, g in zip(client_params, global_parameters)
        ]
        client_deltas.append(delta)

        tau = float(fit_res.metrics.get("local_steps", 1.0))
        tau = max(tau, 1.0)
        client_taus.append(tau)

        p_i = fit_res.num_examples / total_examples
        client_weights.append(p_i)

    tau_eff = sum(p * tau for p, tau in zip(client_weights, client_taus))

    num_layers = len(global_parameters)
    new_params = []

    for layer_idx in range(num_layers):
        g = global_parameters[layer_idx]
        is_buf = (
            buffer_mask[layer_idx]
            if buffer_mask is not None and layer_idx < len(buffer_mask)
            else False
        )
        is_var = (
            var_mask[layer_idx]
            if var_mask is not None and layer_idx < len(var_mask)
            else False
        )

        if is_buf:
            # Standard sample-weighted average for tracking buffers
            agg_layer = sum(
                p_i * np.asarray(c_params[layer_idx], dtype=np.float64)
                for p_i, c_params in zip(client_weights, client_params_list)
            )
            if is_var:
                agg_layer = np.maximum(agg_layer, 0.0)
            new_params.append(agg_layer.astype(g.dtype))
        else:
            # FedNova normalized step aggregation for trainable parameters
            agg_delta = sum(
                p_i * (np.asarray(delta_i[layer_idx], dtype=np.float64) / tau_i)
                for delta_i, p_i, tau_i in zip(client_deltas, client_weights, client_taus)
            )
            agg_delta *= tau_eff
            new_layer = (np.asarray(g, dtype=np.float64) + agg_delta).astype(g.dtype)
            new_params.append(new_layer)

    return new_params


class FedNovaStrategy(FedStrategyBase):
    """FedNova strategy with normalized averaging, extended metrics, and early stopping.

    Args:
        model: Required nn.Module used to inspect layer types (trainable parameters vs BatchNorm buffers).
        early_stop_patience: Number of rounds without improvement before stopping (0 to disable).
        early_stop_metric: Target metric to track ('accuracy' or 'loss').
        metrics_recorder: Optional FLMetricsRecorder instance for logging and visualization.
        local_test: Whether local test partitions are enabled.
        All other kwargs are forwarded to flwr.server.strategy.FedAvg.
    """

    def __init__(
        self,
        *,
        model: Optional[nn.Module] = None,
        early_stop_patience: int = 10,
        early_stop_metric: str = "accuracy",
        metrics_recorder: Optional[FLMetricsRecorder] = None,
        local_test: bool = False,
        **kwargs,
    ):
        if model is None:
            raise ValueError(
                "FedNovaStrategy requires 'model' (nn.Module) to determine layer parameter "
                "types and BatchNorm buffers. Please pass model=<model_instance>."
            )

        super().__init__(
            early_stop_patience=early_stop_patience,
            early_stop_metric=early_stop_metric,
            metrics_recorder=metrics_recorder,
            model=model,
            local_test=local_test,
            strategy_name="FedNova",
            **kwargs,
        )

        self.buffer_mask = get_buffer_mask(model)
        self.var_mask = get_var_buffer_mask(model)
        self._current_global_params: Optional[List[np.ndarray]] = None

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """Record round start time, cache global parameters as ndarrays, or halt if early stopped."""
        ret = super().configure_fit(server_round, parameters, client_manager)
        if ret:
            self._current_global_params = parameters_to_ndarrays(parameters)
        return ret

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """FedNova normalized aggregation with timing metrics."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        if self._current_global_params is not None:
            aggregated_ndarrays = _fednova_aggregate(
                results,
                self._current_global_params,
                buffer_mask=self.buffer_mask,
                var_mask=self.var_mask,
            )
        else:
            from flwr.server.strategy.aggregate import aggregate

            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        local_steps_list = [
            float(fit_res.metrics.get("local_steps", 1))
            for _, fit_res in results
        ]
        metrics_aggregated["local_steps_avg"] = float(np.mean(local_steps_list))
        metrics_aggregated["local_steps_min"] = float(np.min(local_steps_list))
        metrics_aggregated["local_steps_max"] = float(np.max(local_steps_list))

        metrics_aggregated = self._finalize_fit_aggregation(
            server_round=server_round,
            parameters_aggregated=parameters_aggregated,
            metrics_aggregated=metrics_aggregated,
            results=results,
            failures=failures,
        )

        log(
            INFO,
            "[FedNova] Round %d | %.1fs | clients=%d | local_steps=[%.0f-%.0f]",
            server_round,
            metrics_aggregated["round_time_seconds"],
            len(results),
            min(local_steps_list),
            max(local_steps_list),
        )

        return parameters_aggregated, metrics_aggregated
