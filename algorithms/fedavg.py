"""FedAvg Strategy with extended metrics logging, early stopping, and LR scheduling support.

Wraps flwr.server.strategy.FedAvg and FedStrategyBase to add:
- Per-round timing and weight-size tracking.
- Server-side early stopping based on evaluation accuracy or loss.
- Automatic preservation of best global model weights.
- fit_metrics_aggregation_fn that aggregates per-client training metrics.
"""

from logging import INFO
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy

from algorithms.base import (
    FedStrategyBase,
    _format_device_details,
    _weighted_average_metrics,
)
from monitoring.metrics import FLMetricsRecorder


class FedAvgStrategy(FedStrategyBase):
    """FedAvg strategy with extended metrics, early stopping, and per-round timing.

    Args:
        early_stop_patience: Number of rounds without improvement before stopping (0 to disable).
        early_stop_metric: Target metric to track ('accuracy' or 'loss').
        metrics_recorder: Optional FLMetricsRecorder instance for logging and visualization.
        local_test: Whether local test partitions are enabled.
        All other kwargs are forwarded to flwr.server.strategy.FedAvg.
    """

    def __init__(
        self,
        *,
        early_stop_patience: int = 10,
        early_stop_metric: str = "accuracy",
        metrics_recorder: Optional[FLMetricsRecorder] = None,
        local_test: bool = False,
        **kwargs,
    ):
        super().__init__(
            early_stop_patience=early_stop_patience,
            early_stop_metric=early_stop_metric,
            metrics_recorder=metrics_recorder,
            local_test=local_test,
            strategy_name="FedAvg",
            **kwargs,
        )

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

        metrics_aggregated = self._finalize_fit_aggregation(
            server_round=server_round,
            parameters_aggregated=parameters_aggregated,
            metrics_aggregated=metrics_aggregated,
            results=results,
            failures=failures,
        )

        log(
            INFO,
            "[FedAvg] Round %d | %.1fs | clients=%d | failures=%d",
            server_round,
            metrics_aggregated["round_time_seconds"],
            len(results),
            len(failures),
        )

        return parameters_aggregated, metrics_aggregated
