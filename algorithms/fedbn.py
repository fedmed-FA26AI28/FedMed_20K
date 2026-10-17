"""FedBN Strategy - Federated Learning with local Batch Normalization.

Reference:
    Li et al., "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization",
    ICLR 2021. https://arxiv.org/abs/2102.07623

In FedBN, Batch Normalization (BN) layers remain strictly local to each client
to mitigate feature shift across heterogeneous data distributions. Only non-BN
layers (such as Conv2d and Linear weights and biases) are communicated and
aggregated on the server.
"""

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

from algorithms.base import (
    FedStrategyBase,
    _format_device_details,
    _weighted_average_metrics,
)
from models.cnn import get_bn_mask
from monitoring.metrics import FLMetricsRecorder


def _fedbn_aggregate(
    results: List[Tuple[ClientProxy, FitRes]],
    global_parameters: List[np.ndarray],
    bn_mask: List[bool],
) -> List[np.ndarray]:
    """Aggregate model updates by averaging non-BN parameters and preserving server BN parameters."""
    total_examples = sum(fit_res.num_examples for _, fit_res in results)
    if total_examples == 0:
        return global_parameters

    num_layers = len(global_parameters)
    aggregated_params = [np.zeros_like(g, dtype=np.float64) for g in global_parameters]

    client_params_list = []
    client_weights = []
    for _, fit_res in results:
        client_params = parameters_to_ndarrays(fit_res.parameters)
        client_params_list.append(client_params)
        client_weights.append(fit_res.num_examples / total_examples)

    for layer_idx in range(num_layers):
        if layer_idx < len(bn_mask) and bn_mask[layer_idx]:
            # Preserve server global BN parameter (do not aggregate from clients)
            aggregated_params[layer_idx] = np.copy(global_parameters[layer_idx])
        else:
            # Weighted average across clients for non-BN parameters
            for c_params, p_i in zip(client_params_list, client_weights):
                aggregated_params[layer_idx] += p_i * np.asarray(c_params[layer_idx], dtype=np.float64)

    return [
        arr.astype(g.dtype)
        for arr, g in zip(aggregated_params, global_parameters)
    ]


class FedBNStrategy(FedStrategyBase):
    """FedBN Strategy with selective non-BN aggregation, extended metrics, and early stopping.

    Args:
        model: Required PyTorch model instance used to determine BatchNorm layer mask.
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
                "FedBNStrategy requires 'model' (nn.Module) to identify BatchNorm layers "
                "so local BN statistics are not aggregated on the server. Please pass model=<model_instance>."
            )

        super().__init__(
            early_stop_patience=early_stop_patience,
            early_stop_metric=early_stop_metric,
            metrics_recorder=metrics_recorder,
            model=model,
            local_test=local_test,
            strategy_name="FedBN",
            **kwargs,
        )

        self.bn_mask = get_bn_mask(model)
        self.num_bn_layers = sum(self.bn_mask)
        self.num_non_bn_layers = len(self.bn_mask) - self.num_bn_layers
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
        """FedBN aggregation: average non-BN parameters, keep global BN parameters."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        if self._current_global_params is not None:
            aggregated_ndarrays = _fedbn_aggregate(
                results=results,
                global_parameters=self._current_global_params,
                bn_mask=self.bn_mask,
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

        metrics_aggregated["num_bn_layers_local"] = int(self.num_bn_layers)
        metrics_aggregated["num_non_bn_layers_aggregated"] = int(self.num_non_bn_layers)

        metrics_aggregated = self._finalize_fit_aggregation(
            server_round=server_round,
            parameters_aggregated=parameters_aggregated,
            metrics_aggregated=metrics_aggregated,
            results=results,
            failures=failures,
        )

        log(
            INFO,
            "[FedBN] Round %d | %.1fs | clients=%d | non_bn_layers=%d | local_bn_layers=%d",
            server_round,
            metrics_aggregated["round_time_seconds"],
            len(results),
            self.num_non_bn_layers,
            self.num_bn_layers,
        )

        return parameters_aggregated, metrics_aggregated

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        """Configure client evaluation round, passing strategy='fedbn' to preserve local BN."""
        if self.should_stop:
            return []
        evaluate_ins_list = super().configure_evaluate(
            server_round, parameters, client_manager
        )
        for _, evaluate_ins in evaluate_ins_list:
            evaluate_ins.config["strategy"] = "fedbn"
            evaluate_ins.config["server_round"] = server_round
        return evaluate_ins_list

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Centralized server evaluation is disabled in FedBN per official paper."""
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Return a dict summarising the entire FL run."""
        summary = super().get_summary()
        summary["num_bn_layers_local"] = self.num_bn_layers
        summary["num_non_bn_layers_aggregated"] = self.num_non_bn_layers
        return summary

    def save_artifacts(
        self, save_dir: str, extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Export CSVs, comparison plots, and rich fl_results.json."""
        if extra_metadata is None:
            extra_metadata = {}
        extra_metadata["num_bn_layers_local"] = self.num_bn_layers
        extra_metadata["num_non_bn_layers_aggregated"] = self.num_non_bn_layers
        return super().save_artifacts(save_dir=save_dir, extra_metadata=extra_metadata)
