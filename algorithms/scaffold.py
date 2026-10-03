"""SCAFFOLD Strategy - Stochastic Controlled Averaging for Federated Learning.

Reference:
    Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning",
    ICML 2020. https://arxiv.org/abs/1910.06378

SCAFFOLD uses control variates (variance reduction) to correct for "client drift"
caused by non-IID data distributions across clients. The server maintains a global
control variate c, and each client maintains a local control variate c_i.
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
from models.cnn import CNN, get_parameters


class SCAFFOLDStrategy(FlwrFedAvg):
    """SCAFFOLD strategy with control variate aggregation, extended metrics, and early stopping.

    Args:
        num_total_clients: Total number of clients N in the federation (used to scale delta_c).
            Defaults to min_fit_clients.
        model: Optional PyTorch model instance to initialize control variates shape.
        early_stop_patience: Number of rounds without improvement before stopping (0 to disable).
        early_stop_metric: Target metric to track ('accuracy' or 'loss').
        metrics_recorder: Optional FLMetricsRecorder instance for logging and visualization.
        All other kwargs are forwarded to flwr.server.strategy.FedAvg.
    """

    def __init__(
        self,
        *,
        num_total_clients: Optional[int] = None,
        model: Optional[nn.Module] = None,
        early_stop_patience: int = 10,
        early_stop_metric: str = "accuracy",
        metrics_recorder: Optional[FLMetricsRecorder] = None,
        **kwargs,
    ):
        if "fit_metrics_aggregation_fn" not in kwargs:
            kwargs["fit_metrics_aggregation_fn"] = _weighted_average_metrics

        super().__init__(**kwargs)

        self.recorder = metrics_recorder or FLMetricsRecorder(strategy_name="SCAFFOLD")

        self.num_total_clients = num_total_clients or self.min_fit_clients

        # Initialize server control variates c
        target_model = model if model is not None else CNN(num_classes=8)
        initial_ndarrays = get_parameters(target_model)
        self.num_layers = len(initial_ndarrays)
        self.server_control_variate: List[np.ndarray] = [
            np.zeros_like(arr, dtype=np.float64) for arr in initial_ndarrays
        ]

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
        """Append server control variates c to parameters sent to clients for local drift correction."""
        if self.should_stop:
            log(INFO, "[EarlyStop] Training halted: configure_fit sampling 0 clients.")
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

        return super().configure_fit(server_round, combined_parameters, client_manager)

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
        # Using effective client count N = self.num_total_clients
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
        metrics_aggregated["server_c_norm"] = float(c_norm)

        self.recorder.record_fit_results(
            server_round=server_round,
            round_duration=round_time,
            results=results,
            metrics_aggregated=metrics_aggregated,
        )

        log(
            INFO,
            "[SCAFFOLD] Round %d | %.1fs | clients=%d | server_c_norm=%.4f",
            server_round,
            round_time,
            len(results),
            c_norm,
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
        extra_metadata["best_metric"] = self._best_metric
        extra_metadata["best_round"] = self._best_round
        extra_metadata["early_stopped"] = self.should_stop
        extra_metadata["server_c_norm"] = float(
            np.sqrt(sum(np.sum(arr ** 2) for arr in self.server_control_variate))
        )
        return self.recorder.save_all(save_dir=save_dir, extra_metadata=extra_metadata)
