"""FedProx Strategy with extended metrics logging, early stopping, and LR scheduling support.

Wraps flwr.server.strategy.FedProx and adds:
- Per-round timing and weight-size tracking.
- Server-side early stopping based on aggregated evaluation accuracy.
- fit_metrics_aggregation_fn that aggregates per-client training metrics.

FedProx adds a proximal term (mu * ||w - w_global||^2) to the local loss function
to limit how far local models drift from the global model. This is controlled by
the `proximal_mu` parameter sent to clients via on_fit_config_fn.
"""

import time
from typing import Dict, List, Optional, Tuple, Union
from logging import WARNING, INFO

import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedProx as FlwrFedProx

from algorithms.fedavg import _weighted_average_metrics


class FedProxStrategy(FlwrFedProx):
    """FedProx strategy with extended metrics, early stopping, and per-round timing.

    Args:
        proximal_mu: Weight of the proximal term. Higher values keep local models
            closer to the global model. Typical range: 0.001 – 1.0.
        early_stop_patience: Number of rounds without improvement before stopping.
            Set to 0 to disable server-side early stopping.
        early_stop_metric: Metric key returned by evaluate to track (default: accuracy).
        All other kwargs are forwarded to flwr.server.strategy.FedProx.
    """

    def __init__(
        self,
        *,
        proximal_mu: float = 0.1,
        early_stop_patience: int = 10,
        early_stop_metric: str = "accuracy",
        **kwargs,
    ):
        # Inject our custom metrics aggregation unless caller overrides
        if "fit_metrics_aggregation_fn" not in kwargs:
            kwargs["fit_metrics_aggregation_fn"] = _weighted_average_metrics

        # FedProx requires proximal_mu to be passed
        super().__init__(proximal_mu=proximal_mu, **kwargs)

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

        round_time = time.time() - self._round_start
        self._round_times.append(round_time)

        metrics_aggregated["round_time_seconds"] = float(round_time)
        metrics_aggregated["total_elapsed_seconds"] = float(
            time.time() - self._total_start
        )
        metrics_aggregated["num_clients_reporting"] = len(results)
        metrics_aggregated["num_failures"] = len(failures)

        log(
            INFO,
            "[FedProx] Round %d | %.1fs | clients=%d | failures=%d | mu=%.4f",
            server_round,
            round_time,
            len(results),
            len(failures),
            self.proximal_mu,
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
    #  Summary helper
    # ------------------------------------------------------------------ #

    def get_summary(self) -> Dict[str, float]:
        """Return a dict summarising the entire FL run."""
        total_time = time.time() - self._total_start
        return {
            "strategy": "FedProx",
            "proximal_mu": self.proximal_mu,
            "total_rounds": len(self._round_times),
            "total_time_seconds": total_time,
            "avg_round_time_seconds": float(np.mean(self._round_times))
            if self._round_times
            else 0.0,
            "best_metric": self._best_metric if self._best_metric else 0.0,
        }
