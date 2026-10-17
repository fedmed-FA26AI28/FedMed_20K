"""SCAFFOLD Strategy - Stochastic Controlled Averaging for Federated Learning.

Reference:
    Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning",
    ICML 2020. https://arxiv.org/abs/1910.06378

SCAFFOLD uses control variates (variance reduction) to correct for "client drift"
caused by non-IID data distributions across clients. The server maintains a global
control variate c, and each client maintains a local control variate c_i.
"""

import time
from logging import INFO, WARNING
from typing import Dict, List, Optional, Tuple, Union, Any

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
from flwr.server.strategy import FedAvg as FlwrFedAvg

from algorithms.base import (
    FedStrategyBase,
    _format_device_details,
    _weighted_average_metrics,
)
from models.cnn import get_parameters
from monitoring.metrics import FLMetricsRecorder


class SCAFFOLDStrategy(FedStrategyBase):
    """SCAFFOLD strategy with control variate aggregation, extended metrics, and early stopping.

    Args:
        model: Required PyTorch model instance to initialize control variates shape.
        num_total_clients: Total number of clients N in the federation (used to scale delta_c).
            Defaults to min_fit_clients.
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
        num_total_clients: Optional[int] = None,
        early_stop_patience: int = 10,
        early_stop_metric: str = "accuracy",
        metrics_recorder: Optional[FLMetricsRecorder] = None,
        local_test: bool = False,
        **kwargs,
    ):
        if model is None:
            raise ValueError(
                "SCAFFOLDStrategy requires 'model' (nn.Module) to initialize server control "
                "variates with the exact model parameter shapes and layer counts. "
                "Please pass model=<model_instance>."
            )

        super().__init__(
            early_stop_patience=early_stop_patience,
            early_stop_metric=early_stop_metric,
            metrics_recorder=metrics_recorder,
            model=model,
            local_test=local_test,
            strategy_name="SCAFFOLD",
            **kwargs,
        )

        self.num_total_clients = num_total_clients or self.min_fit_clients

        # Initialize server control variates c matching model architecture
        initial_ndarrays = get_parameters(model)
        self.num_layers = len(initial_ndarrays)
        self.server_control_variate: List[np.ndarray] = [
            np.zeros_like(arr, dtype=np.float64) for arr in initial_ndarrays
        ]
        self._current_global_params: Optional[List[np.ndarray]] = None

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """Append server control variates c to parameters sent to clients for local drift correction."""
        if self.should_stop:
            log(INFO, "[SCAFFOLD] EarlyStop: Training halted: configure_fit sampling 0 clients.")
            return []

        self._round_start = time.time()
        self.recorder.record_round_start(server_round)

        # Cache standard model parameters (M ndarrays)
        model_ndarrays = parameters_to_ndarrays(parameters)
        self._current_global_params = model_ndarrays
        self.num_layers = len(model_ndarrays)

        # Ensure server control variate matches parameter shapes
        if len(self.server_control_variate) != self.num_layers:
            self.server_control_variate = [
                np.zeros_like(p, dtype=np.float64) for p in model_ndarrays
            ]

        # Concatenate model parameters x and server control variates c (2M ndarrays total)
        combined_ndarrays = model_ndarrays + self.server_control_variate
        combined_parameters = ndarrays_to_parameters(combined_ndarrays)

        return super(FedStrategyBase, self).configure_fit(
            server_round, combined_parameters, client_manager
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """SCAFFOLD aggregation: average model weights and update global control variate c."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        if total_examples == 0:
            return None, {}

        # Unpack each client's payload: [y_i (M arrays)] + [delta_c_i (M arrays)]
        client_models: List[List[np.ndarray]] = []
        client_delta_cs: List[List[np.ndarray]] = []
        client_weights: List[float] = []

        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            if len(arrays) == 2 * self.num_layers:
                y_i = arrays[: self.num_layers]
                delta_c_i = arrays[self.num_layers :]
            else:
                # Fallback if client only returned model parameters
                y_i = arrays
                delta_c_i = [np.zeros_like(a, dtype=np.float64) for a in arrays]

            client_models.append(y_i)
            client_delta_cs.append(delta_c_i)
            client_weights.append(fit_res.num_examples / total_examples)

        # 1. Aggregate model parameters x_new = sum(p_i * y_i)
        aggregated_model = [np.zeros_like(g, dtype=np.float64) for g in self._current_global_params]
        for y_i, p_i in zip(client_models, client_weights):
            for l_idx in range(self.num_layers):
                aggregated_model[l_idx] += p_i * np.asarray(y_i[l_idx], dtype=np.float64)

        # Cast back to original layer dtypes (e.g. integer buffers like num_batches_tracked)
        final_model = [
            arr.astype(g.dtype)
            for arr, g in zip(aggregated_model, self._current_global_params)
        ]

        # 2. Update server control variate c_new = c + (1 / N) * sum(delta_c_i)
        n_clients = max(self.num_total_clients, len(results))
        for delta_c_i in client_delta_cs:
            for l_idx in range(self.num_layers):
                self.server_control_variate[l_idx] = self.server_control_variate[l_idx] + (
                    1.0 / n_clients
                ) * np.asarray(delta_c_i[l_idx], dtype=np.float64)

        # Compute norm of server control variates for telemetry
        c_norm = float(
            np.sqrt(sum(np.sum(arr ** 2) for arr in self.server_control_variate))
        )

        parameters_aggregated = ndarrays_to_parameters(final_model)

        metrics_aggregated: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        metrics_aggregated["server_c_norm"] = float(c_norm)

        metrics_aggregated = self._finalize_fit_aggregation(
            server_round=server_round,
            parameters_aggregated=parameters_aggregated,
            metrics_aggregated=metrics_aggregated,
            results=results,
            failures=failures,
        )

        log(
            INFO,
            "[SCAFFOLD] Round %d | %.1fs | clients=%d | server_c_norm=%.4f",
            server_round,
            metrics_aggregated["round_time_seconds"],
            len(results),
            c_norm,
        )

        return parameters_aggregated, metrics_aggregated

    def get_summary(self) -> Dict[str, Any]:
        """Return a dict summarising the entire FL run."""
        summary = super().get_summary()
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
        extra_metadata["server_c_norm"] = float(
            np.sqrt(sum(np.sum(arr ** 2) for arr in self.server_control_variate))
        )
        return super().save_artifacts(save_dir=save_dir, extra_metadata=extra_metadata)
