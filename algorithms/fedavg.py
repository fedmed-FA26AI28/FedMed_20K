"""FedAvg Strategy with extended metrics logging, early stopping, and LR scheduling support.

Wraps flwr.server.strategy.FedAvg and adds:
- Per-round timing and weight-size tracking.
- Server-side early stopping based on evaluation accuracy or loss.
- Automatic preservation of best global model weights.
- fit_metrics_aggregation_fn that aggregates per-client training metrics.
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
from flwr.server.strategy.aggregate import aggregate

from monitoring.metrics import FLMetricsRecorder


def _weighted_average_metrics(
    metrics: List[Tuple[int, Dict[str, Scalar]]],
) -> Dict[str, Scalar]:
    """Aggregate fit metrics from all clients using weighted average."""
    if not metrics:
        return {}

    total_examples = sum(n for n, _ in metrics)
    aggregated: Dict[str, Scalar] = {}

    epoch_times = []
    training_times = []
    weight_sizes_kb = []
    ping_times = []

    for n, m in metrics:
        for key in ("train_loss", "train_accuracy"):
            if key in m:
                aggregated[key] = aggregated.get(key, 0.0) + float(m[key]) * n

        if "epoch_time_avg" in m:
            epoch_times.append(float(m["epoch_time_avg"]))
        if "training_time" in m:
            training_times.append(float(m["training_time"]))
        if "weight_size_kb" in m:
            weight_sizes_kb.append(float(m["weight_size_kb"]))
        if "ping_ms" in m and float(m["ping_ms"]) >= 0:
            ping_times.append(float(m["ping_ms"]))

    # Finalize weighted averages
    for key in ("train_loss", "train_accuracy"):
        if key in aggregated:
            aggregated[key] = float(aggregated[key]) / total_examples

    # Per-client timing stats
    if epoch_times:
        aggregated["epoch_time_avg"] = float(np.mean(epoch_times))
        aggregated["epoch_time_min"] = float(np.min(epoch_times))
        aggregated["epoch_time_max"] = float(np.max(epoch_times))

    if training_times:
        aggregated["client_train_time_avg"] = float(np.mean(training_times))
        aggregated["client_train_time_min"] = float(np.min(training_times))
        aggregated["client_train_time_max"] = float(np.max(training_times))
        aggregated["straggler_time"] = float(np.max(training_times))

    if weight_sizes_kb:
        aggregated["weight_size_kb_avg"] = float(np.mean(weight_sizes_kb))
        aggregated["weight_size_kb_total"] = float(np.sum(weight_sizes_kb))

    if ping_times:
        aggregated["ping_ms_avg"] = float(np.mean(ping_times))
        aggregated["ping_ms_min"] = float(np.min(ping_times))
        aggregated["ping_ms_max"] = float(np.max(ping_times))

    return aggregated


def _format_device_details(results: List[Tuple[ClientProxy, FitRes]]) -> str:
    """Format participating client devices with client_id, device_ipv4, and device_type."""
    details = []
    for _, fit_res in results:
        cid = fit_res.metrics.get("client_id")
        ip = fit_res.metrics.get("device_ipv4")
        dtype = fit_res.metrics.get("device_type")
        if cid is not None or ip is not None:
            details.append(f"Client {cid} (IPv4: {ip or 'unknown'}, {dtype or 'unknown'})")
    return " | ".join(details) if details else ""


class FedAvgStrategy(FlwrFedAvg):
    """FedAvg strategy with extended metrics, early stopping, and per-round timing.

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

        self.recorder = metrics_recorder or FLMetricsRecorder(strategy_name="FedAvg")

        # Early stopping state
        self.early_stop_patience = early_stop_patience
        self.early_stop_metric = early_stop_metric
        self._best_metric: Optional[float] = None
        self._best_round: int = 0
        self._best_parameters: Optional[Parameters] = None
        self._rounds_without_improvement: int = 0
        self._last_checked_round: Optional[int] = None
        self.should_stop: bool = False
        self.latest_parameters: Optional[Parameters] = None

        # Per-round timing
        self._round_start: float = 0.0
        self._round_times: List[float] = []
        self._total_start: float = time.time()

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

        if self._last_checked_round == server_round:
            return self.should_stop
        self._last_checked_round = server_round

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
        """Record round start time, then delegate to parent or halt if early stopped."""
        if self.should_stop:
            log(INFO, "[EarlyStop] Training halted: configure_fit returning 0 clients.")
            return []
        self._round_start = time.time()
        self.recorder.record_round_start(server_round)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client parameters and append round timing metrics."""
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )
        if parameters_aggregated is not None:
            self.latest_parameters = parameters_aggregated

        round_time = time.time() - self._round_start
        self._round_times.append(round_time)

        metrics_aggregated["round_time_seconds"] = float(round_time)
        metrics_aggregated["total_elapsed_seconds"] = float(
            time.time() - self._total_start
        )
        metrics_aggregated["num_clients_reporting"] = len(results)
        metrics_aggregated["num_failures"] = len(failures)

        self.recorder.record_fit_results(
            server_round=server_round,
            round_duration=round_time,
            results=results,
            metrics_aggregated=metrics_aggregated,
        )

        log(
            INFO,
            "[FedAvg] Round %d | %.1fs | clients=%d | failures=%d",
            server_round,
            round_time,
            len(results),
            len(failures),
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

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):
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
