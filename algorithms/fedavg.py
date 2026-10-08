"""FedAvg Strategy with extended metrics logging, early stopping, and LR scheduling support.

Wraps flwr.server.strategy.FedAvg and adds:
- Per-round timing and weight-size tracking.
- Server-side early stopping based on aggregated evaluation accuracy.
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


def _weighted_average_metrics(
    metrics: List[Tuple[int, Dict[str, Scalar]]],
) -> Dict[str, Scalar]:
    """Aggregate fit metrics from all clients using weighted average."""
    if not metrics:
        return {}

    total_examples = sum(n for n, _ in metrics)

    aggregated: Dict[str, Scalar] = {}
    total_val_examples = 0

    # --- Per-client epoch time ---
    epoch_times = []
    training_times = []
    weight_sizes_kb = []
    ping_times = []

    for n, m in metrics:
        # Weighted loss and accuracy
        for key in ("train_loss", "train_accuracy", "distill_loss"):
            if key in m:
                aggregated[key] = aggregated.get(key, 0.0) + float(m[key]) * n

        num_val_examples = int(m.get("num_val_samples", 0))
        if num_val_examples > 0:
            total_val_examples += num_val_examples
            for key in ("val_loss", "val_accuracy"):
                if key in m:
                    aggregated[key] = (
                        aggregated.get(key, 0.0)
                        + float(m[key]) * num_val_examples
                    )

        if "epoch_time_avg" in m:
            epoch_times.append(float(m["epoch_time_avg"]))
        if "training_time" in m:
            training_times.append(float(m["training_time"]))
        if "weight_size_kb" in m:
            weight_sizes_kb.append(float(m["weight_size_kb"]))
        if "ping_ms" in m and float(m["ping_ms"]) >= 0:
            ping_times.append(float(m["ping_ms"]))

    # Finalize weighted averages
    for key in ("train_loss", "train_accuracy", "distill_loss"):
        if key in aggregated:
            aggregated[key] = float(aggregated[key]) / total_examples
    for key in ("val_loss", "val_accuracy"):
        if key in aggregated and total_val_examples > 0:
            aggregated[key] = float(aggregated[key]) / total_val_examples

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


from algorithms.fedavg import _weighted_average_metrics
from monitoring.metrics import FLMetricsRecorder


class FedAvgStrategy(FlwrFedAvg):
    """FedAvg strategy with extended metrics, early stopping, and per-round timing.

    Args:
        early_stop_patience: Number of rounds without improvement before stopping.
            Set to 0 to disable server-side early stopping.
        early_stop_metric: Metric key returned by evaluate to track (default: accuracy).
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
        # Inject our custom metrics aggregation unless caller overrides
        if "fit_metrics_aggregation_fn" not in kwargs:
            kwargs["fit_metrics_aggregation_fn"] = _weighted_average_metrics

        super().__init__(**kwargs)

        # Metrics recorder
        self.recorder = metrics_recorder or FLMetricsRecorder(strategy_name="FedAvg")

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

    # ------------------------------------------------------------------ #
    #  Timing hooks
    # ------------------------------------------------------------------ #

    def configure_fit(self, server_round, parameters, client_manager):
        """Record round start time, then delegate to parent."""
        self._round_start = time.time()
        self.recorder.record_round_start(server_round)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate and append round-level timing metrics."""
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )
        if parameters_aggregated is not None:
            # Metrics are logged separately; the final global model always comes
            # from aggregated client parameters weighted by training samples.
            self.latest_parameters = parameters_aggregated

        round_time = time.time() - self._round_start
        self._round_times.append(round_time)

        metrics_aggregated["round_time_seconds"] = float(round_time)
        metrics_aggregated["total_elapsed_seconds"] = float(
            time.time() - self._total_start
        )
        metrics_aggregated["num_clients_reporting"] = len(results)
        metrics_aggregated["num_failures"] = len(failures)

        # Record in comprehensive metrics recorder
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

        return parameters_aggregated, metrics_aggregated

    # ------------------------------------------------------------------ #
    #  Early stopping in aggregate_evaluate
    # ------------------------------------------------------------------ #

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

        # Check early stopping only if patience > 0
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
                        "[EarlyStop] Round %d: no improvement for %d rounds (best=%.4f, current=%.4f)",
                        server_round,
                        self._rounds_without_improvement,
                        self._best_metric,
                        current,
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
