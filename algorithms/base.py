"""Base Federated Learning Strategy for FedMedAI.

Provides a unified base class (FedStrategyBase) wrapping Flower's FedAvg strategy
with shared functionality:
- Client fit metrics aggregation and device telemetry formatting.
- Server-side early stopping based on evaluation accuracy or loss.
- Automatic preservation of best global model weights.
- Round timing, straggler tracking, and parameter dynamics recording.
- Dual-mode evaluation tracking (Local Test MCTA/BMCTA/BTA vs Global Test).
- Unified artifact and summary reporting.
"""

import time
from logging import INFO, WARNING
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch.nn as nn
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg as FlwrFedAvg

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
            aggregated[key] = float(aggregated[key]) / max(total_examples, 1)

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


class FedStrategyBase(FlwrFedAvg):
    """Base strategy class encapsulating common FL execution, monitoring, and early stopping.

    Args:
        early_stop_patience: Number of rounds without improvement before stopping (0 to disable).
        early_stop_metric: Target metric to track ('accuracy' or 'loss').
        metrics_recorder: Optional FLMetricsRecorder instance for logging and visualization.
        model: Optional nn.Module instance for architecture and parameter inspections.
        local_test: Whether local test partitions are enabled.
        strategy_name: Name of the strategy for reporting.
        **kwargs: Forwarded to flwr.server.strategy.FedAvg.
    """

    def __init__(
        self,
        *,
        early_stop_patience: int = 10,
        early_stop_metric: str = "accuracy",
        metrics_recorder: Optional[FLMetricsRecorder] = None,
        model: Optional[nn.Module] = None,
        local_test: bool = False,
        strategy_name: str = "FedAvg",
        **kwargs,
    ):
        if "fit_metrics_aggregation_fn" not in kwargs:
            kwargs["fit_metrics_aggregation_fn"] = _weighted_average_metrics

        super().__init__(**kwargs)

        self.strategy_name = strategy_name
        self.model = model
        self.recorder = metrics_recorder or FLMetricsRecorder(strategy_name=strategy_name)
        self.local_test = local_test
        self.recorder.local_test_enabled = local_test

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
        self._current_global_params: Optional[Any] = None
        self._last_finalized_fit_round: Optional[int] = None

    def check_early_stopping(
        self,
        server_round: int,
        metrics: Dict[str, Any],
        loss: Optional[float] = None,
        parameters: Optional[Parameters] = None,
    ) -> bool:
        """Evaluate early stopping conditions and track best model parameters.
        
        Best parameter checkpointing is decoupled from early-stopping patience:
        _best_metric, _best_round, and _best_parameters are continuously tracked
        across all rounds even when early_stop_patience <= 0 (disabled).
        """
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
                    self._best_parameters = Parameters(
                        tensors=list(parameters.tensors),
                        tensor_type=parameters.tensor_type,
                    )
                log(
                    INFO,
                    "[BestModel] Round %d: new best %s = %.4f",
                    server_round,
                    self.early_stop_metric,
                    current,
                )
            else:
                self._rounds_without_improvement += 1
                if self.early_stop_patience > 0:
                    log(
                        INFO,
                        "[EarlyStop] Round %d: no improvement for %d/%d rounds (best=%.4f, current=%.4f)",
                        server_round,
                        self._rounds_without_improvement,
                        self.early_stop_patience,
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

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """Record round start time, then delegate to parent or halt if early stopped."""
        if self.should_stop:
            log(INFO, f"[{self.strategy_name}] EarlyStop: Training halted: configure_fit returning 0 clients.")
            return []
        self._round_start = time.time()
        self._current_global_params = parameters
        self.recorder.record_round_start(server_round)
        return super().configure_fit(server_round, parameters, client_manager)

    def _ensure_recorder_client_stores(self) -> None:
        """Ensure recorder containers for per-client trajectories exist."""
        for attr in ("client_full_histories", "client_class_distributions", "client_eval_histories"):
            if not hasattr(self.recorder, attr) or getattr(self.recorder, attr) is None:
                setattr(self.recorder, attr, {})

    @staticmethod
    def _merge_round_records(
        existing: List[Dict[str, Any]],
        incoming: List[Dict[str, Any]],
        round_keys: Tuple[str, ...] = ("server_round", "round"),
    ) -> List[Dict[str, Any]]:
        """Merge round dictionaries in ascending round order, preserving existing non-None keys."""
        by_round: Dict[int, Dict[str, Any]] = {}
        for item in list(existing or []) + list(incoming or []):
            if not isinstance(item, dict):
                continue
            r_val = None
            for rk in round_keys:
                if item.get(rk) is not None:
                    try:
                        r_val = int(item[rk])
                        break
                    except (TypeError, ValueError):
                        pass
            if r_val is None:
                continue
            cur = dict(by_round.get(r_val, {}))
            for k, v in item.items():
                if v is not None:
                    cur[k] = v
            by_round[r_val] = cur
        return [by_round[r] for r in sorted(by_round.keys())]

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: Any,
    ) -> List[Tuple[ClientProxy, Any]]:
        """Inject strategy metadata and server_round into client evaluation instructions."""
        client_instructions = super().configure_evaluate(server_round, parameters, client_manager)
        strategy_label = getattr(
            self,
            "strategy_name",
            getattr(self.recorder, "strategy_name", "fedavg"),
        )
        for _, evaluate_ins in client_instructions:
            if evaluate_ins.config is None:
                evaluate_ins.config = {}
            evaluate_ins.config.setdefault("strategy", strategy_label)
            evaluate_ins.config["server_round"] = int(server_round)
        return client_instructions

    def _finalize_fit_aggregation(
        self,
        server_round: int,
        parameters_aggregated: Optional[Parameters],
        metrics_aggregated: Dict[str, Scalar],
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Dict[str, Scalar]:
        """Record common parameter dynamics, timings, and reporting metrics across all strategies."""
        if getattr(self, "_last_finalized_fit_round", None) == server_round:
            return metrics_aggregated
        self._last_finalized_fit_round = server_round

        import json

        self._ensure_recorder_client_stores()

        # Emit real-time per-round client metrics before heavier dynamics/logging work
        for _, fit_res in results:
            m = fit_res.metrics or {}
            cid = m.get("client_id")
            if cid is not None:
                try:
                    int_cid = int(cid)
                    known_dist = self.recorder.client_class_distributions.get(int_cid, {})
                    round_pkg: Dict[str, Any] = {
                        "client_id": int_cid,
                        "server_round": int(server_round),
                        "train_loss": float(m.get("train_loss", 0.0)),
                        "train_accuracy": float(m.get("train_accuracy", 0.0)),
                        "round_time": float(m.get("round_time", m.get("training_time", 0.0))),
                        "epoch_time_avg": float(m.get("epoch_time_avg", 0.0)),
                        "ping_ms": float(m.get("ping_ms", -1.0)),
                        "weight_size_kb": float(m.get("weight_size_kb", 0.0)),
                        "num_samples": int(fit_res.num_examples),
                        "device_ipv4": str(m.get("device_ipv4", "")),
                        "device_type": str(m.get("device_type", "")),
                    }
                    if "eval_loss" in m and m["eval_loss"] is not None:
                        round_pkg["eval_loss"] = float(m["eval_loss"])
                    if "eval_accuracy" in m and m["eval_accuracy"] is not None:
                        round_pkg["eval_accuracy"] = float(m["eval_accuracy"])
                    if known_dist:
                        round_pkg["distribution"] = known_dist

                    fit_entry = {
                        "round": int(server_round),
                        "server_round": int(server_round),
                        "train_loss": round_pkg["train_loss"],
                        "train_accuracy": round_pkg["train_accuracy"],
                        "round_time": round_pkg["round_time"],
                        "epoch_time_avg": round_pkg["epoch_time_avg"],
                        "ping_ms": round_pkg["ping_ms"],
                        "weight_size_kb": round_pkg["weight_size_kb"],
                        "num_samples": round_pkg["num_samples"],
                    }
                    self.recorder.client_full_histories[int_cid] = self._merge_round_records(
                        self.recorder.client_full_histories.get(int_cid, []),
                        [fit_entry],
                        round_keys=("round", "server_round"),
                    )
                    if "eval_loss" in round_pkg or "eval_accuracy" in round_pkg:
                        eval_entry: Dict[str, Any] = {"server_round": int(server_round)}
                        if "eval_loss" in round_pkg:
                            eval_entry["eval_loss"] = round_pkg["eval_loss"]
                        if "eval_accuracy" in round_pkg:
                            eval_entry["eval_accuracy"] = round_pkg["eval_accuracy"]
                        self.recorder.client_eval_histories[int_cid] = self._merge_round_records(
                            self.recorder.client_eval_histories.get(int_cid, []),
                            [eval_entry],
                            round_keys=("server_round", "round"),
                        )

                    print(f"[CLIENT_ROUND_METRICS] {json.dumps(round_pkg)}", flush=True)
                except Exception as e:
                    log(WARNING, "[Server] Failed to emit client round metrics for client %s: %s", str(cid), e)

        if parameters_aggregated is not None:
            if hasattr(self.recorder, "record_parameter_dynamics"):
                prev_params = getattr(self, "_current_global_params", None)
                self.recorder.record_parameter_dynamics(
                    server_round=server_round,
                    previous_parameters=prev_params,
                    aggregated_parameters=parameters_aggregated,
                    results=results,
                )
            self.latest_parameters = parameters_aggregated

        round_time = time.time() - self._round_start
        self._round_times.append(round_time)

        metrics_aggregated["round_time_seconds"] = float(round_time)
        metrics_aggregated["total_elapsed_seconds"] = float(time.time() - self._total_start)
        metrics_aggregated["num_clients_reporting"] = len(results)
        metrics_aggregated["num_failures"] = len(failures)

        self.recorder.record_fit_results(
            server_round=server_round,
            round_duration=round_time,
            results=results,
            metrics_aggregated=metrics_aggregated,
        )

        # Unpack final-round complete client metrics & data distribution bundles if present
        for _, fit_res in results:
            m = fit_res.metrics or {}
            cid = m.get("client_id")
            if "final_client_history_json" in m and cid is not None:
                try:
                    hist = json.loads(m["final_client_history_json"])
                    class_dist = json.loads(m.get("class_counts_json", "{}"))
                    eval_hist = json.loads(m.get("eval_history_json", "[]"))

                    int_cid = int(cid)
                    merged_hist = self._merge_round_records(
                        self.recorder.client_full_histories.get(int_cid, []),
                        hist,
                        round_keys=("round", "server_round"),
                    )
                    self.recorder.client_full_histories[int_cid] = merged_hist

                    if isinstance(class_dist, dict) and class_dist:
                        self.recorder.client_class_distributions[int_cid] = class_dist
                    known_dist = self.recorder.client_class_distributions.get(int_cid, {})

                    merged_eval_hist = self._merge_round_records(
                        self.recorder.client_eval_histories.get(int_cid, []),
                        eval_hist,
                        round_keys=("server_round", "round"),
                    )
                    self.recorder.client_eval_histories[int_cid] = merged_eval_hist

                    summary_pkg = {
                        "client_id": int_cid,
                        "rounds": len(merged_hist),
                        "samples": int(fit_res.num_examples),
                        "distribution": known_dist,
                        "history": merged_hist,
                        "eval_history": merged_eval_hist,
                    }
                    print(f"[CLIENT_FINAL_METRICS] {json.dumps(summary_pkg)}", flush=True)
                    log(INFO, "[Server] Received complete post-run metrics & class distribution from Client %d", int_cid)
                except Exception as e:
                    log(WARNING, "[Server] Failed to decode final client metrics package for client %s: %s", str(cid), e)

        dev_str = _format_device_details(results)
        if dev_str:
            log(INFO, "[Reporting Devices] %s", dev_str)

        return metrics_aggregated

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using standard FedAvg and finalize recording metrics."""
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )
        if parameters_aggregated is not None or metrics_aggregated:
            metrics_aggregated = self._finalize_fit_aggregation(
                server_round=server_round,
                parameters_aggregated=parameters_aggregated,
                metrics_aggregated=metrics_aggregated,
                results=results,
                failures=failures,
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

    def _record_and_emit_client_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, Any]],
    ) -> None:
        """Record per-client evaluation history and stream live eval updates to the GUI."""
        if not results:
            return
        import json

        self._ensure_recorder_client_stores()
        for _, eval_res in results:
            num_examples = int(getattr(eval_res, "num_examples", 0) or 0)
            if num_examples <= 0:
                continue
            m = getattr(eval_res, "metrics", None) or {}
            cid = m.get("client_id")
            if cid is None:
                continue
            try:
                int_cid = int(cid)
                eval_loss = (
                    float(m["eval_loss"])
                    if m.get("eval_loss") is not None
                    else (float(eval_res.loss) if getattr(eval_res, "loss", None) is not None else None)
                )
                acc_raw = m.get("eval_accuracy", m.get("accuracy"))
                eval_acc = float(acc_raw) if acc_raw is not None else None
                if eval_loss is None and eval_acc is None:
                    continue

                if m.get("class_counts_json"):
                    try:
                        parsed_dist = json.loads(m["class_counts_json"])
                        if isinstance(parsed_dist, dict) and parsed_dist:
                            self.recorder.client_class_distributions[int_cid] = parsed_dist
                    except Exception:
                        pass
                known_dist = self.recorder.client_class_distributions.get(int_cid, {})

                eval_entry: Dict[str, Any] = {"server_round": int(server_round)}
                if eval_loss is not None:
                    eval_entry["eval_loss"] = eval_loss
                if eval_acc is not None:
                    eval_entry["eval_accuracy"] = eval_acc

                merged_eval = self._merge_round_records(
                    self.recorder.client_eval_histories.get(int_cid, []),
                    [eval_entry],
                    round_keys=("server_round", "round"),
                )
                self.recorder.client_eval_histories[int_cid] = merged_eval

                round_pkg: Dict[str, Any] = {
                    "client_id": int_cid,
                    "server_round": int(server_round),
                }
                if eval_loss is not None:
                    round_pkg["eval_loss"] = eval_loss
                if eval_acc is not None:
                    round_pkg["eval_accuracy"] = eval_acc
                if m.get("num_samples") is not None:
                    round_pkg["num_samples"] = int(m["num_samples"])
                if known_dist:
                    round_pkg["distribution"] = known_dist

                print(f"[CLIENT_ROUND_METRICS] {json.dumps(round_pkg)}", flush=True)

                if int_cid in self.recorder.client_full_histories:
                    full_hist = self.recorder.client_full_histories.get(int_cid, [])
                    samples_val = int(
                        m.get("num_samples")
                        if m.get("num_samples") is not None
                        else (full_hist[-1].get("num_samples", 0) if full_hist else 0)
                    )
                    summary_pkg = {
                        "client_id": int_cid,
                        "rounds": max(len(full_hist), len(merged_eval)),
                        "samples": samples_val,
                        "distribution": known_dist,
                        "history": full_hist,
                        "eval_history": merged_eval,
                    }
                    print(f"[CLIENT_FINAL_METRICS] {json.dumps(summary_pkg)}", flush=True)
            except Exception as e:
                log(WARNING, "[Server] Failed to emit client eval metrics for client %s: %s", str(cid), e)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, Any]],
        failures: List[Union[Tuple[ClientProxy, Any], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results and check early stopping condition."""
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )

        self._record_and_emit_client_evaluate(server_round, results)

        if results and hasattr(self.recorder, "record_federated_eval_results"):
            self.recorder.record_federated_eval_results(
                server_round=server_round,
                results=results,
                metrics_aggregated=metrics_aggregated or {},
                local_test=self.local_test,
            )
            comb_acc = metrics_aggregated.get("accuracy", 0.0) if metrics_aggregated else 0.0
            loss_prefix = f"Loss: {float(loss_aggregated):.4f} | " if loss_aggregated is not None else ""
            if self.local_test:
                mcta_val = self.recorder.mcta_history.get(server_round, 0.0)
                bmcta_val = self.recorder.bmcta
                bta_val = self.recorder.bta
                log(
                    INFO,
                    "[%s] Round %d Eval | %sMCTA: %.2f%% (BMCTA: %.2f%% @ R%d) | Combined: %.2f%% (BTA: %.2f%% @ R%d)",
                    self.strategy_name,
                    server_round,
                    loss_prefix,
                    mcta_val * 100,
                    bmcta_val * 100,
                    self.recorder.bmcta_round,
                    float(comb_acc) * 100,
                    bta_val * 100,
                    self.recorder.bta_round,
                )
            else:
                log(
                    INFO,
                    "[%s] Round %d Eval | %sGlobal Test Acc: %.2f%% (Best: %.2f%% @ R%d)",
                    self.strategy_name,
                    server_round,
                    loss_prefix,
                    float(comb_acc) * 100,
                    self.recorder.bta * 100,
                    self.recorder.bta_round,
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

        if metrics_aggregated:
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
