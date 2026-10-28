"""Coverage + LA Federated Learning Strategy.

Combines two client-side mechanisms with standard FedAvg aggregation:
- Coverage head penalty: anchors classifier head rows to the global reference,
  weighted inversely by per-class local sample count.
- Logit Adjustment (LA): adds log(n_{k,c} + epsilon) offsets to logits before
  cross-entropy during training.

Server-side aggregation is unchanged from FedAvg (weighted average by n_k).
All Coverage + LA computation happens locally on each client.

Reference: coverage_LA.md (project documentation).
"""

from logging import INFO
from typing import Any, Dict, List, Optional, Tuple, Union

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


class CoverageLA_Strategy(FedStrategyBase):
    """Coverage + Logit Adjustment strategy with FedAvg aggregation.

    Client-side training uses:
    - Coverage head penalty (mu_head, kappa) to anchor the classifier head.
    - Logit Adjustment (tau, epsilon) to shift logits by log sample counts.

    Server-side aggregation is standard FedAvg weighted by client sample counts.

    Args:
        coverage_kappa: Positive float controlling coverage weight curve (default: 512).
        coverage_mu_head: Non-negative float for head penalty intensity (default: 0.1).
        la_tau: Non-negative float for logit adjustment intensity (default: 1.0).
        la_epsilon: Positive float for label smoothing in log counts (default: 1.0).
        early_stop_patience: Rounds without improvement before stopping (0 = disabled).
        early_stop_metric: Target metric to track ('accuracy' or 'loss').
        metrics_recorder: Optional FLMetricsRecorder instance.
        local_test: Whether local test partitions are enabled.
        **kwargs: Forwarded to flwr.server.strategy.FedAvg.
    """

    def __init__(
        self,
        *,
        coverage_kappa: float = 512.0,
        coverage_mu_head: float = 0.1,
        la_tau: float = 1.0,
        la_epsilon: float = 1.0,
        early_stop_patience: int = 10,
        early_stop_metric: str = "accuracy",
        metrics_recorder: Optional[FLMetricsRecorder] = None,
        local_test: bool = False,
        **kwargs,
    ):
        # Validate parameters per document constraints
        if coverage_kappa <= 0:
            raise ValueError(f"coverage_kappa must be > 0, got {coverage_kappa}")
        if coverage_mu_head < 0:
            raise ValueError(f"coverage_mu_head must be >= 0, got {coverage_mu_head}")
        if la_tau < 0:
            raise ValueError(f"la_tau must be >= 0, got {la_tau}")
        if la_epsilon <= 0:
            raise ValueError(f"la_epsilon must be > 0, got {la_epsilon}")

        self.coverage_kappa = float(coverage_kappa)
        self.coverage_mu_head = float(coverage_mu_head)
        self.la_tau = float(la_tau)
        self.la_epsilon = float(la_epsilon)

        recorder = metrics_recorder or FLMetricsRecorder(
            strategy_name="CoverageLA",
        )

        super().__init__(
            early_stop_patience=early_stop_patience,
            early_stop_metric=early_stop_metric,
            metrics_recorder=recorder,
            local_test=local_test,
            strategy_name="CoverageLA",
            **kwargs,
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client parameters using standard FedAvg weighted average."""
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
            "[CoverageLA] Round %d | %.1fs | clients=%d | failures=%d | "
            "kappa=%.1f mu_head=%.4f tau=%.4f eps=%.4f",
            server_round,
            metrics_aggregated["round_time_seconds"],
            len(results),
            len(failures),
            self.coverage_kappa,
            self.coverage_mu_head,
            self.la_tau,
            self.la_epsilon,
        )

        return parameters_aggregated, metrics_aggregated

    def get_summary(self) -> Dict[str, Any]:
        """Return run summary including Coverage + LA hyperparameters."""
        summary = super().get_summary()
        summary["coverage_kappa"] = self.coverage_kappa
        summary["coverage_mu_head"] = self.coverage_mu_head
        summary["la_tau"] = self.la_tau
        summary["la_epsilon"] = self.la_epsilon
        return summary


__all__ = ["CoverageLA_Strategy"]
