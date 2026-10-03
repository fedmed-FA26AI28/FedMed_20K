"""FedBN Strategy - Federated Learning with local Batch Normalization.

Reference:
    Li et al., "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization",
    ICLR 2021. https://arxiv.org/abs/2102.07623

In FedBN, Batch Normalization (BN) layers remain strictly local to each client
to mitigate feature shift across heterogeneous data distributions. Only non-BN
layers (such as Conv2d and Linear weights and biases) are communicated and
aggregated on the server.
"""

import time
from typing import Dict, List, Optional, Tuple, Union, Any
from logging import WARNING, INFO

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

from algorithms.fedavg import _weighted_average_metrics, _format_device_details
from monitoring.metrics import FLMetricsRecorder
from models.cnn import CNN, get_bn_mask


def _fedbn_aggregate(
    results: List[Tuple[ClientProxy, FitRes]],
    global_parameters: List[np.ndarray],
    bn_mask: List[bool],
) -> List[np.ndarray]:
    """Aggregate model updates by averaging non-BN parameters and preserving server BN parameters.

    Args:
        results: List of (ClientProxy, FitRes) from reporting clients.
        global_parameters: Current global parameters on the server.
        bn_mask: Boolean list matching layers, True if layer belongs to BatchNorm.

    Returns:
        List of NumPy ndarrays representing updated global model parameters.
    """
    total_examples = sum(fit_res.num_examples for _, fit_res in results)
    if total_examples == 0:
        return global_parameters

    num_layers = len(global_parameters)
    aggregated_params = [np.zeros_like(g, dtype=np.float64) for g in global_parameters]

    # Pre-extract client parameter ndarrays and normalized weights
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


class FedBNStrategy(FlwrFedAvg):
    """FedBN Strategy with selective non-BN aggregation, extended metrics, and early stopping.

    Args:
        model: Optional PyTorch model instance used to determine BatchNorm layer mask.
            Defaults to CNN(num_classes=8).
        early_stop_patience: Number of rounds without improvement before stopping (0 to disable).
        early_stop_metric: Target metric to track ('accuracy' or 'loss').
        metrics_recorder: Optional FLMetricsRecorder instance for logging and visualization.
        All other kwargs are forwarded to flwr.server.strategy.FedAvg.
    """

    def __init__(
        self,
        *,
        model: Optional[nn.Module] = None,
        early_stop_patience: int = 10,
        early_stop_metric: str = "accuracy",
        metrics_recorder: Optional[FLMetricsRecorder] = None,
        **kwargs,
    ):
        if "fit_metrics_aggregation_fn" not in kwargs:
            kwargs["fit_metrics_aggregation_fn"] = _weighted_average_metrics

        super().__init__(**kwargs)

        self.recorder = metrics_recorder or FLMetricsRecorder(strategy_name="FedBN")

        # Resolve BatchNorm parameter mask
        target_model = model if model is not None else CNN(num_classes=8)
        self.bn_mask = get_bn_mask(target_model)
        self.num_bn_layers = sum(self.bn_mask)
        self.num_non_bn_layers = len(self.bn_mask) - self.num_bn_layers

        # Early stopping state
        self.early_stop_patience = early_stop_patience
        self.early_stop_metric = early_stop_metric
        self._best_metric: Optional[float] = None
        self._best_round: int = 0
        self._best_parameters: Optional[Parameters] = None
        self._rounds_without_improvement: int = 0
        self.should_stop: bool = False
        self.latest_parameters: Optional[Parameters] = None

        # Per-round timing
        self._round_start: float = 0.0
        self._round_times: List[float] = []
        self._total_start: float = time.time()

        self._current_global_params: Optional[List[np.ndarray]] = None

    def check_early_stopping(
        self,
        server_round: int,
        metrics: Dict[str, Any],
        loss: Optional[float] = None,
        parameters: Optional[Parameters] = None,
    ) -> bool:
        """Evaluate early stopping conditions against monitored metric."""
        if self.early_stop_patience <= 0:
            return False

        if self.early_stop_metric == "loss":
            current = loss if loss is not None else metrics.get("loss")
            is_better = lambda cur, best: best is None or cur < (best - 1e-5)
        else:
            current = metrics.get(self.early_stop_metric)
            is_better = lambda cur, best: best is None or cur > (best + 1e-5)

        if current is not None:
            current = float(current)
            if is_better(current, self._best_metric):
                self._best_metric = current
                self._best_round = server_round
                self._rounds_without_improvement = 0
                if parameters is not None:
                    self._best_parameters = parameters
                log(
                    INFO,
                    "[EarlyStop] Round %d: new best %s = %.4f",
                    server_round,
                    self.early_stop_metric,
                    current,
                )
            else:
                self._rounds_without_improvement += 1
                log(
                    INFO,
                    "[EarlyStop] Round %d: no improvement for %d rounds (best=%.4f, current=%.4f)",
                    server_round,
                    self._rounds_without_improvement,
                    self._best_metric if self._best_metric is not None else 0.0,
                    current,
                )
                if self._rounds_without_improvement >= self.early_stop_patience:
                    log(
                        WARNING,
                        "[EarlyStop] Stopping at round %d - no improvement in %d rounds.",
                        server_round,
                        self.early_stop_patience,
                    )
                    self.should_stop = True
                    return True
        return self.should_stop

    def get_best_parameters(self) -> Optional[Parameters]:
        """Return the best recorded model parameters or latest if none."""
        return self._best_parameters or self.latest_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        """Record round start time, cache global parameters, or halt if early stopped."""
        if self.should_stop:
            log(INFO, "[EarlyStop] Training halted: configure_fit sampling 0 clients.")
            return []
        self._round_start = time.time()
        self.recorder.record_round_start(server_round)
        self._current_global_params = parameters_to_ndarrays(parameters)
        return super().configure_fit(server_round, parameters, client_manager)

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
        self.latest_parameters = parameters_aggregated

        metrics_aggregated: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        round_time = time.time() - self._round_start
        self._round_times.append(round_time)

        metrics_aggregated["round_time_seconds"] = float(round_time)
        metrics_aggregated["total_elapsed_seconds"] = float(
            time.time() - self._total_start
        )
        metrics_aggregated["num_clients_reporting"] = len(results)
        metrics_aggregated["num_failures"] = len(failures)
        metrics_aggregated["num_bn_layers_local"] = int(self.num_bn_layers)
        metrics_aggregated["num_non_bn_layers_aggregated"] = int(self.num_non_bn_layers)

        self.recorder.record_fit_results(
            server_round=server_round,
            round_duration=round_time,
            results=results,
            metrics_aggregated=metrics_aggregated,
        )

        log(
            INFO,
            "[FedBN] Round %d | %.1fs | clients=%d | non_bn_layers=%d | local_bn_layers=%d",
            server_round,
            round_time,
            len(results),
            self.num_non_bn_layers,
            self.num_bn_layers,
        )
        dev_str = _format_device_details(results)
        if dev_str:
            log(INFO, "[Reporting Devices] %s", dev_str)

        return parameters_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using evaluate_fn and check early stopping."""
        res = super().evaluate(server_round, parameters)
        if res is not None:
            loss, metrics = res
            self.check_early_stopping(
                server_round=server_round,
                metrics=metrics,
                loss=loss,
                parameters=parameters,
            )
        return res

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results and check early stopping condition."""
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )

        if loss_aggregated is not None and metrics_aggregated:
            eval_metric_val = metrics_aggregated.get(self.early_stop_metric)
            if eval_metric_val is not None:
                self.recorder.record_eval_results(
                    server_round=server_round,
                    loss=loss_aggregated,
                    accuracy=float(eval_metric_val),
                    is_server_eval=False,
                )

        if self.early_stop_patience > 0 and metrics_aggregated:
            self.check_early_stopping(
                server_round=server_round,
                metrics=metrics_aggregated,
                loss=loss_aggregated,
                parameters=self.latest_parameters,
            )

        return loss_aggregated, metrics_aggregated

    def get_summary(self) -> Dict[str, Any]:
        """Return a dict summarising the entire FL run."""
        summary = self.recorder.get_summary()
        summary["best_metric"] = self._best_metric
        summary["best_round"] = self._best_round
        summary["early_stopped"] = self.should_stop
        summary["num_bn_layers_local"] = self.num_bn_layers
        summary["num_non_bn_layers_aggregated"] = self.num_non_bn_layers
        return summary

    def save_artifacts(
        self, save_dir: str, extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Export CSVs, comparison plots, and rich fl_results.json."""
        if extra_metadata is None:
            extra_metadata = {}
        extra_metadata["best_metric"] = self._best_metric
        extra_metadata["best_round"] = self._best_round
        extra_metadata["early_stopped"] = self.should_stop
        extra_metadata["num_bn_layers_local"] = self.num_bn_layers
        extra_metadata["num_non_bn_layers_aggregated"] = self.num_non_bn_layers
        return self.recorder.save_all(save_dir=save_dir, extra_metadata=extra_metadata)
