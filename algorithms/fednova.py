"""FedNova Strategy - Normalized Averaging for heterogeneous client compute.

FedNova adjusts the aggregation weights by the number of local steps taken
by each client, eliminating objective inconsistency caused by heterogeneous
local updates (e.g., fast clients taking more steps than slow clients).
"""

import time
from typing import Dict, List, Optional, Tuple, Union, Any
from logging import WARNING, INFO

import numpy as np
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

from algorithms.fedavg import _weighted_average_metrics
from monitoring.metrics import FLMetricsRecorder


def _fednova_aggregate(
    results: List[Tuple[ClientProxy, FitRes]],
    global_parameters: List[np.ndarray],
) -> List[np.ndarray]:
    """Aggregate model updates using FedNova normalized averaging.

    Each client i computes:
        delta_i = w_i - w_global
        d_i = delta_i / tau_i  (normalized direction, tau_i = local_steps)

    Effective number of steps:
        tau_eff = sum_i(p_i * tau_i)  where p_i = n_i / n_total

    Aggregated update:
        delta_nova = tau_eff * sum_i(p_i * d_i)
        w_new = w_global + delta_nova
    """
    total_examples = sum(fit_res.num_examples for _, fit_res in results)
    if total_examples == 0:
        return global_parameters

    client_deltas = []
    client_weights = []
    client_taus = []

    for _, fit_res in results:
        client_params = parameters_to_ndarrays(fit_res.parameters)
        delta = [c - g for c, g in zip(client_params, global_parameters)]
        client_deltas.append(delta)

        tau = float(fit_res.metrics.get("local_steps", 1.0))
        tau = max(tau, 1.0)
        client_taus.append(tau)

        p_i = fit_res.num_examples / total_examples
        client_weights.append(p_i)

    tau_eff = sum(p * tau for p, tau in zip(client_weights, client_taus))

    num_layers = len(global_parameters)
    aggregated_delta = [np.zeros_like(g) for g in global_parameters]

    for layer_idx in range(num_layers):
        for delta_i, p_i, tau_i in zip(client_deltas, client_weights, client_taus):
            d_i = delta_i[layer_idx] / tau_i
            aggregated_delta[layer_idx] += p_i * d_i
        aggregated_delta[layer_idx] *= tau_eff

    new_params = [
        global_parameters[idx] + aggregated_delta[idx]
        for idx in range(len(global_parameters))
    ]

    return new_params


class FedNovaStrategy(FlwrFedAvg):
    """FedNova strategy with normalized averaging, extended metrics, and early stopping.

    Args:
        early_stop_patience: Number of rounds without improvement before stopping (0 to disable).
        early_stop_metric: Target metric to track ('accuracy' or 'loss').
        metrics_recorder: Optional FLMetricsRecorder instance for logging and visualization.
        All other kwargs are forwarded to flwr.server.strategy.FedAvg.
    """

    def __init__(
        self,
        *,
        early_stop_patience: int = 10,
        early_stop_metric: str = "accuracy",
        metrics_recorder: Optional[FLMetricsRecorder] = None,
        **kwargs,
    ):
        if "fit_metrics_aggregation_fn" not in kwargs:
            kwargs["fit_metrics_aggregation_fn"] = _weighted_average_metrics

        super().__init__(**kwargs)

        self.recorder = metrics_recorder or FLMetricsRecorder(strategy_name="FedNova")

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
        """FedNova normalized aggregation with timing metrics."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        if self._current_global_params is not None:
            aggregated_ndarrays = _fednova_aggregate(
                results, self._current_global_params
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

        local_steps_list = [
            float(fit_res.metrics.get("local_steps", 1))
            for _, fit_res in results
        ]
        metrics_aggregated["local_steps_avg"] = float(np.mean(local_steps_list))
        metrics_aggregated["local_steps_min"] = float(np.min(local_steps_list))
        metrics_aggregated["local_steps_max"] = float(np.max(local_steps_list))

        self.recorder.record_fit_results(
            server_round=server_round,
            round_duration=round_time,
            results=results,
            metrics_aggregated=metrics_aggregated,
        )

        log(
            INFO,
            "[FedNova] Round %d | %.1fs | clients=%d | local_steps=[%.0f-%.0f]",
            server_round,
            round_time,
            len(results),
            min(local_steps_list),
            max(local_steps_list),
        )

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
        return self.recorder.save_all(save_dir=save_dir, extra_metadata=extra_metadata)
