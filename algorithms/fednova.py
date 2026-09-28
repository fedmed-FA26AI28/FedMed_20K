"""FedNova Strategy — Normalized Averaging for heterogeneous local training.

FedNova (Wang et al., 2020) addresses objective inconsistency in FedAvg
when clients perform different numbers of local steps. Instead of simple
weighted averaging, FedNova normalizes each client's update by its
number of local gradient steps (tau_i).

Aggregation formula:
    d_global = sum_i (p_i * d_i / tau_i) * tau_eff
    w_new    = w_global - d_global

where:
    p_i      = n_i / N  (data ratio)
    d_i      = w_global - w_i_local  (pseudo-gradient)
    tau_i    = local steps of client i
    tau_eff  = sum_i (p_i * tau_i)  (effective number of steps)

Clients send `local_steps` in their metrics dict.
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
    """FedNova normalized aggregation.

    Each client's update is normalised by its number of local steps (tau_i).
    Clients MUST report 'local_steps' in fit_res.metrics.
    Falls back to standard weighted averaging if local_steps is missing.
    """
    total_examples = sum(fit_res.num_examples for _, fit_res in results)

    # Collect per-client data ratios and local steps
    p_values: List[float] = []
    tau_values: List[float] = []
    client_params_list: List[List[np.ndarray]] = []

    for _, fit_res in results:
        p_i = fit_res.num_examples / total_examples
        tau_i = float(fit_res.metrics.get("local_steps", 1))
        p_values.append(p_i)
        tau_values.append(tau_i)
        client_params_list.append(parameters_to_ndarrays(fit_res.parameters))

    # Effective number of steps
    tau_eff = sum(p * t for p, t in zip(p_values, tau_values))

    # Compute normalized aggregate
    # d_i = w_global - w_i  (pseudo-gradient, note the sign)
    aggregated = [np.zeros_like(g) for g in global_parameters]

    for p_i, tau_i, client_params in zip(p_values, tau_values, client_params_list):
        for layer_idx in range(len(global_parameters)):
            d_i = global_parameters[layer_idx] - client_params[layer_idx]
            aggregated[layer_idx] += (p_i / tau_i) * d_i

    # w_new = w_global - tau_eff * aggregated_normalized_gradient
    new_params = [
        global_parameters[idx] - tau_eff * aggregated[idx]
        for idx in range(len(global_parameters))
    ]

    return new_params


class FedNovaStrategy(FlwrFedAvg):
    """FedNova strategy with normalized averaging, extended metrics, and early stopping.

    Args:
        early_stop_patience: Number of rounds without improvement before stopping.
            Set to 0 to disable.
        early_stop_metric: Metric key to track (default: accuracy).
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

        # Metrics recorder
        self.recorder = metrics_recorder or FLMetricsRecorder(strategy_name="FedNova")

        # Early stopping state
        self.early_stop_patience = early_stop_patience
        self.early_stop_metric = early_stop_metric
        self._best_metric: Optional[float] = None
        self._rounds_without_improvement: int = 0
        self.should_stop: bool = False

        # Per-round timing
        self._round_start: float = 0.0
        self._round_times: List[float] = []
        self._total_start: float = time.time()

        # Store the current global parameters for FedNova delta computation
        self._current_global_params: Optional[List[np.ndarray]] = None

    # ------------------------------------------------------------------ #
    #  Timing hooks
    # ------------------------------------------------------------------ #

    def configure_fit(self, server_round, parameters, client_manager):
        """Record round start time and cache global parameters for delta computation."""
        self._round_start = time.time()
        self.recorder.record_round_start(server_round)
        # Cache current global parameters for use in aggregate_fit
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

        # --- FedNova aggregation ---
        if self._current_global_params is not None:
            aggregated_ndarrays = _fednova_aggregate(
                results, self._current_global_params
            )
        else:
            # First round fallback: standard weighted average
            from flwr.server.strategy.aggregate import aggregate

            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics
        metrics_aggregated: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # Timing
        round_time = time.time() - self._round_start
        self._round_times.append(round_time)

        metrics_aggregated["round_time_seconds"] = float(round_time)
        metrics_aggregated["total_elapsed_seconds"] = float(
            time.time() - self._total_start
        )
        metrics_aggregated["num_clients_reporting"] = len(results)
        metrics_aggregated["num_failures"] = len(failures)

        # Log local_steps from each client
        local_steps_list = [
            float(fit_res.metrics.get("local_steps", 1))
            for _, fit_res in results
        ]
        metrics_aggregated["local_steps_avg"] = float(np.mean(local_steps_list))
        metrics_aggregated["local_steps_min"] = float(np.min(local_steps_list))
        metrics_aggregated["local_steps_max"] = float(np.max(local_steps_list))

        # Record in comprehensive metrics recorder
        self.recorder.record_fit_results(
            server_round=server_round,
            round_duration=round_time,
            results=results,
            metrics_aggregated=metrics_aggregated,
        )

        log(
            INFO,
            "[FedNova] Round %d | %.1fs | clients=%d | local_steps=[%.0f–%.0f]",
            server_round,
            round_time,
            len(results),
            min(local_steps_list),
            max(local_steps_list),
        )

        return parameters_aggregated, metrics_aggregated

    # ------------------------------------------------------------------ #
    #  Early stopping in aggregate_evaluate
    # ------------------------------------------------------------------ #

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
            current = metrics_aggregated.get(self.early_stop_metric)
            if current is not None:
                current = float(current)
                if self._best_metric is None or current > self._best_metric:
                    self._best_metric = current
                    self._rounds_without_improvement = 0
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
                        "[EarlyStop] Round %d: no improvement for %d rounds",
                        server_round,
                        self._rounds_without_improvement,
                    )
                    if self._rounds_without_improvement >= self.early_stop_patience:
                        log(
                            WARNING,
                            "[EarlyStop] Stopping at round %d — no improvement in %d rounds.",
                            server_round,
                            self.early_stop_patience,
                        )
                        self.should_stop = True

        return loss_aggregated, metrics_aggregated

    # ------------------------------------------------------------------ #
    #  Summary & Artifacts
    # ------------------------------------------------------------------ #

    def get_summary(self) -> Dict[str, Any]:
        """Return a dict summarising the entire FL run."""
        return self.recorder.get_summary()

    def save_artifacts(
        self, save_dir: str, extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Export CSVs, comparison plots, and rich fl_results.json."""
        return self.recorder.save_all(save_dir=save_dir, extra_metadata=extra_metadata)
