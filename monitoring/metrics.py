"""Federated Learning Metrics Recorder and Visualizer for FedMedAI.

Captures, processes, and exports:
1. Client-level and round-level training metrics to structured CSV files:
   - client_round_metrics.csv: Per-client round-by-round time, loss, accuracy, resource stats.
   - client_epoch_metrics.csv: Granular per-epoch execution time, loss, accuracy per client.
   - round_metrics.csv: Server-level round latency, straggler delay, communication volume, global metrics.
2. Publication-grade comparison plots:
   - client_training_time_comparison.png: Per-client training times & straggler latency per round.
   - client_epoch_time_comparison.png: Epoch execution times across clients and device classes.
   - fl_training_curves.png: Convergence curves (global eval vs client training).
   - round_time_breakdown.png: Round duration decomposition (compute, straggler wait, comm overhead).
   - fl_summary_card.png: Visual summary dashboard card for the FL run.
3. Enhanced, deeply informative JSON report (fl_results.json):
   - Full experiment metadata, executive summary stats, per-client statistics,
     hardware heterogeneity analysis, round-by-round trajectory, and artifact links.
"""

import os
import json
import time
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np

# Use headless backend for matplotlib before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", font="sans-serif")
except ImportError:
    sns = None

try:
    import pandas as pd
except ImportError:
    pd = None

from monitoring.resource import get_device_type

try:
    from flwr.common import parameters_to_ndarrays, Parameters
except ImportError:
    parameters_to_ndarrays = None
    Parameters = None


def _to_ndarrays(params: Any) -> Optional[List[np.ndarray]]:
    """Safely convert Flower Parameters or array sequences into List[np.ndarray]."""
    if params is None:
        return None
    if isinstance(params, (list, tuple)):
        return [np.asarray(p) for p in params]
    if parameters_to_ndarrays is not None:
        try:
            return parameters_to_ndarrays(params)
        except Exception:
            pass
    return None


def parse_json_or_list(val: Any) -> List[float]:
    """Safely parse JSON-encoded string, comma-separated string, or list into List[float]."""
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return [float(x) for x in val if x is not None]
    if isinstance(val, (int, float)):
        return [float(val)]
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return []
        if val.startswith("[") and val.endswith("]"):
            try:
                parsed = json.loads(val)
                return [float(x) for x in parsed if x is not None]
            except Exception:
                pass
        # Try comma-delimited
        try:
            return [float(x.strip()) for x in val.split(",") if x.strip()]
        except Exception:
            pass
    return []


class FLMetricsRecorder:
    """Central metrics collector and report generator for FedMedAI FL runs."""

    def __init__(
        self,
        strategy_name: str = "FedAvg",
        proximal_mu: Optional[float] = None,
        model_name: Optional[str] = None,
        model: Optional[Any] = None,
        total_parameters: Optional[int] = None,
        trainable_parameters: Optional[int] = None,
        model_weight_size_kb: Optional[float] = None,
        model_file_size_kb: Optional[float] = None,
    ):
        self.strategy_name = strategy_name
        self.proximal_mu = proximal_mu
        self.fl_start_time = time.time()

        # Model architecture & weight size metadata
        self.model_name: Optional[str] = model_name
        self.total_parameters: Optional[int] = total_parameters
        self.trainable_parameters: Optional[int] = trainable_parameters
        self.model_weight_size_kb: Optional[float] = model_weight_size_kb
        self.model_file_size_kb: Optional[float] = model_file_size_kb

        if model is not None or model_name is not None:
            self.set_model_info(
                model_name=model_name,
                model=model,
                total_parameters=total_parameters,
                trainable_parameters=trainable_parameters,
                model_weight_size_kb=model_weight_size_kb,
                model_file_size_kb=model_file_size_kb,
            )

        self.round_start_times: Dict[int, float] = {}
        self.round_records: List[Dict[str, Any]] = []
        self.client_records: List[Dict[str, Any]] = []
        self.epoch_records: List[Dict[str, Any]] = []
        self.eval_records: Dict[int, Dict[str, Any]] = {}

        self.best_eval_accuracy: float = 0.0
        self.best_eval_round: int = 0
        self.best_eval_loss: float = float("inf")

        # FedSLD Paper Metrics: BMCTA, BMCTA Loss, and BTA Tracking
        self.local_test_enabled: bool = False
        self.mcta_history: Dict[int, float] = {}
        self.wcta_history: Dict[int, float] = {}
        self.mctl_history: Dict[int, float] = {}
        self.client_eval_records: List[Dict[str, Any]] = []
        self.bmcta: float = 0.0
        self.bmcta_round: int = 0
        self.bmcta_loss: Optional[float] = None
        self.bta: float = 0.0
        self.bta_round: int = 0

        self.parameter_round_records: List[Dict[str, Any]] = []
        self.parameter_client_records: List[Dict[str, Any]] = []
        self.parameter_layer_records: List[Dict[str, Any]] = []

    def set_model_info(
        self,
        model_name: Optional[str] = None,
        model: Optional[Any] = None,
        total_parameters: Optional[int] = None,
        trainable_parameters: Optional[int] = None,
        model_weight_size_kb: Optional[float] = None,
        model_file_size_kb: Optional[float] = None,
    ) -> None:
        """Register or compute model architecture metadata, parameter count, and weight size footprint."""
        if model_name is not None:
            self.model_name = str(model_name)
        if model is not None:
            try:
                self.total_parameters = sum(p.numel() for p in model.parameters())
                self.trainable_parameters = sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                )
                param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
                self.model_weight_size_kb = param_bytes / 1024.0

                import io
                import torch
                buf = io.BytesIO()
                torch.save(model.state_dict(), buf)
                self.model_file_size_kb = len(buf.getvalue()) / 1024.0
            except Exception as e:
                print(f"[MetricsRecorder] Warning computing model info from model instance: {e}")
        elif self.model_name is not None and self.total_parameters is None:
            try:
                from models.factory import get_model
                temp_m = get_model(self.model_name)
                self.total_parameters = sum(p.numel() for p in temp_m.parameters())
                self.trainable_parameters = sum(
                    p.numel() for p in temp_m.parameters() if p.requires_grad
                )
                param_bytes = sum(p.numel() * p.element_size() for p in temp_m.parameters())
                self.model_weight_size_kb = param_bytes / 1024.0
                import io
                import torch
                buf = io.BytesIO()
                torch.save(temp_m.state_dict(), buf)
                self.model_file_size_kb = len(buf.getvalue()) / 1024.0
            except Exception:
                pass

        if total_parameters is not None:
            self.total_parameters = int(total_parameters)
        if trainable_parameters is not None:
            self.trainable_parameters = int(trainable_parameters)
        if model_weight_size_kb is not None:
            self.model_weight_size_kb = float(model_weight_size_kb)
        if model_file_size_kb is not None:
            self.model_file_size_kb = float(model_file_size_kb)

    # ───────────────────────────────────────────────────────────
    # Recording Hooks
    # ───────────────────────────────────────────────────────────

    def record_round_start(self, server_round: int):
        """Record start timestamp for a server round."""
        self.round_start_times[server_round] = time.time()

    def record_fit_results(
        self,
        server_round: int,
        round_duration: float,
        results: List[Tuple[Any, Any]],
        metrics_aggregated: Dict[str, Any],
    ):
        """Record per-client fit outputs and calculate round-level stats."""
        round_clients: List[Dict[str, Any]] = []

        for client_proxy, fit_res in results:
            m = fit_res.metrics or {}

            # Extract client identity
            raw_cid = m.get("client_id")
            if raw_cid is None:
                try:
                    raw_cid = int(client_proxy.cid)
                except Exception:
                    raw_cid = len(round_clients)
            client_id = int(raw_cid)

            device_type = str(m.get("device_type") or get_device_type(client_id))
            num_samples = int(fit_res.num_examples)
            local_epochs = int(m.get("local_epochs") or m.get("epochs_run") or 1)
            learning_rate = float(m.get("learning_rate", 0.001))

            training_time = float(m.get("training_time", round_duration))
            epoch_time_avg = float(
                m.get("epoch_time_avg", training_time / max(1, local_epochs))
            )
            epoch_time_min = float(m.get("epoch_time_min", epoch_time_avg))
            epoch_time_max = float(m.get("epoch_time_max", epoch_time_avg))

            train_loss = float(m.get("train_loss", 0.0))
            train_accuracy = float(m.get("train_accuracy", 0.0))
            weight_size_kb = float(m.get("weight_size_kb", 0.0))

            cpu_percent = float(m.get("cpu_percent", 0.0))
            ram_percent = float(m.get("ram_percent", 0.0))
            ram_used_mb = float(m.get("ram_used_mb", 0.0))
            gpu_memory_mb = float(m.get("gpu_memory_mb", 0.0))

            # Ping latency
            raw_ping = m.get("ping_ms")
            ping_ms = float(raw_ping) if raw_ping is not None and float(raw_ping) >= 0 else None

            epoch_times = parse_json_or_list(m.get("epoch_times"))
            if not epoch_times:
                epoch_times = [round(epoch_time_avg, 4)] * local_epochs

            epoch_losses = parse_json_or_list(m.get("epoch_losses"))
            if not epoch_losses:
                epoch_losses = [round(train_loss, 4)] * len(epoch_times)

            epoch_accuracies = parse_json_or_list(m.get("epoch_accuracies"))
            if not epoch_accuracies:
                epoch_accuracies = [round(train_accuracy, 4)] * len(epoch_times)

            epoch_lrs = parse_json_or_list(m.get("epoch_lrs"))
            if not epoch_lrs:
                epoch_lrs = [learning_rate] * len(epoch_times)

            client_entry = {
                "round": server_round,
                "client_id": client_id,
                "device_type": device_type,
                "num_samples": num_samples,
                "local_epochs": local_epochs,
                "learning_rate": learning_rate,
                "training_time_seconds": round(training_time, 4),
                "epoch_time_avg": round(epoch_time_avg, 4),
                "epoch_time_min": round(epoch_time_min, 4),
                "epoch_time_max": round(epoch_time_max, 4),
                "train_loss": round(train_loss, 6),
                "train_accuracy": round(train_accuracy, 6),
                "ping_ms": round(ping_ms, 2) if ping_ms is not None else None,
                "weight_size_kb": round(weight_size_kb, 2),
                "cpu_percent": round(cpu_percent, 2),
                "ram_percent": round(ram_percent, 2),
                "ram_used_mb": round(ram_used_mb, 2),
                "gpu_memory_mb": round(gpu_memory_mb, 2),
                "epoch_times": epoch_times,
                "epoch_losses": epoch_losses,
                "epoch_accuracies": epoch_accuracies,
                "epoch_lrs": epoch_lrs,
            }
            round_clients.append(client_entry)
            self.client_records.append(client_entry)

            # Record granular epoch-level records
            cum_time = 0.0
            for ep_idx, ep_time in enumerate(epoch_times):
                cum_time += ep_time
                ep_loss = epoch_losses[ep_idx] if ep_idx < len(epoch_losses) else train_loss
                ep_acc = (
                    epoch_accuracies[ep_idx]
                    if ep_idx < len(epoch_accuracies)
                    else train_accuracy
                )
                ep_lr = epoch_lrs[ep_idx] if ep_idx < len(epoch_lrs) else learning_rate

                self.epoch_records.append(
                    {
                        "round": server_round,
                        "client_id": client_id,
                        "device_type": device_type,
                        "epoch": ep_idx + 1,
                        "epoch_time_seconds": round(ep_time, 4),
                        "cumulative_epoch_time_seconds": round(cum_time, 4),
                        "train_loss": round(ep_loss, 6),
                        "train_accuracy": round(ep_acc, 6),
                        "learning_rate": ep_lr,
                    }
                )

        # Compute round-level statistics
        client_train_times = [c["training_time_seconds"] for c in round_clients]
        c_min = min(client_train_times) if client_train_times else 0.0
        c_max = max(client_train_times) if client_train_times else 0.0
        c_avg = float(np.mean(client_train_times)) if client_train_times else 0.0
        straggler_time = max(0.0, c_max - c_min)
        server_overhead = max(0.0, round_duration - c_max)

        train_loss_avg = float(
            metrics_aggregated.get(
                "train_loss",
                np.mean([c["train_loss"] for c in round_clients])
                if round_clients
                else 0.0,
            )
        )
        train_acc_avg = float(
            metrics_aggregated.get(
                "train_accuracy",
                np.mean([c["train_accuracy"] for c in round_clients])
                if round_clients
                else 0.0,
            )
        )
        epoch_time_avg = float(
            np.mean([c["epoch_time_avg"] for c in round_clients])
            if round_clients
            else 0.0
        )
        total_weight_kb = float(sum(c["weight_size_kb"] for c in round_clients))

        eval_info = self.eval_records.get(server_round, {})
        eval_loss = eval_info.get("eval_loss")
        eval_acc = eval_info.get("eval_accuracy")

        total_elapsed = time.time() - self.fl_start_time

        # Round-level ping stats
        round_pings = [
            c["ping_ms"] for c in round_clients if c.get("ping_ms") is not None
        ]
        ping_ms_avg = float(np.mean(round_pings)) if round_pings else None
        ping_ms_min = float(np.min(round_pings)) if round_pings else None
        ping_ms_max = float(np.max(round_pings)) if round_pings else None

        round_entry = {
            "round": server_round,
            "round_time_seconds": round(round_duration, 4),
            "total_elapsed_seconds": round(total_elapsed, 4),
            "num_clients_reporting": len(round_clients),
            "train_loss_avg": round(train_loss_avg, 6),
            "train_accuracy_avg": round(train_acc_avg, 6),
            "eval_loss": round(eval_loss, 6) if eval_loss is not None else None,
            "eval_accuracy": round(eval_acc, 6) if eval_acc is not None else None,
            "mcta": round(self.mcta_history.get(server_round, 0.0), 6) if server_round in self.mcta_history else None,
            "mctl": round(self.mctl_history.get(server_round, 0.0), 6) if server_round in self.mctl_history else None,
            "bmcta": round(self.bmcta, 6) if self.bmcta > 0 else None,
            "bmcta_round": self.bmcta_round if self.bmcta > 0 else None,
            "bmcta_loss": round(self.bmcta_loss, 6) if self.bmcta_loss is not None else None,
            "wcta": round(self.wcta_history.get(server_round, 0.0), 6) if server_round in self.wcta_history else None,
            "bta": round(self.bta, 6) if self.bta > 0 else None,
            "client_train_time_avg": round(c_avg, 4),
            "client_train_time_min": round(c_min, 4),
            "client_train_time_max": round(c_max, 4),
            "straggler_time_seconds": round(straggler_time, 4),
            "epoch_time_avg": round(epoch_time_avg, 4),
            "total_weight_size_kb": round(total_weight_kb, 2),
            "server_overhead_seconds": round(server_overhead, 4),
            "ping_ms_avg": round(ping_ms_avg, 2) if ping_ms_avg is not None else None,
            "ping_ms_min": round(ping_ms_min, 2) if ping_ms_min is not None else None,
            "ping_ms_max": round(ping_ms_max, 2) if ping_ms_max is not None else None,
        }
        self.round_records.append(round_entry)

    def record_eval_results(
        self,
        server_round: int,
        loss: float,
        accuracy: float,
        is_server_eval: bool = False,
    ):
        """Record evaluation loss and accuracy for a round."""
        loss_val = float(loss) if loss is not None else None
        acc_val = float(accuracy) if accuracy is not None else None

        self.eval_records[server_round] = {
            "eval_loss": loss_val,
            "eval_accuracy": acc_val,
            "is_server_eval": is_server_eval,
        }

        if acc_val is not None:
            if acc_val > self.best_eval_accuracy:
                self.best_eval_accuracy = acc_val
                self.best_eval_round = server_round
                if loss_val is not None:
                    self.best_eval_loss = loss_val
            if acc_val > self.bta:
                self.bta = acc_val
                self.bta_round = server_round

        # Update existing round record if already present
        for r in self.round_records:
            if r["round"] == server_round:
                r["eval_loss"] = round(loss_val, 6) if loss_val is not None else None
                r["eval_accuracy"] = round(acc_val, 6) if acc_val is not None else None
                r["bta"] = round(self.bta, 6) if self.bta > 0 else None
                break

    def record_federated_eval_results(
        self,
        server_round: int,
        results: List[Tuple[Any, Any]],
        metrics_aggregated: Dict[str, Any],
        local_test: bool = False,
    ):
        """Record federated client evaluation results, compute MCTA, WCTA, MCTL, and track BMCTA/BTA/BMCTA Loss."""
        self.local_test_enabled = bool(local_test)
        client_eval_entries = []
        client_accuracies = []
        client_losses = []

        for client_proxy, eval_res in results:
            m = eval_res.metrics or {}
            raw_cid = m.get("client_id")
            if raw_cid is None:
                try:
                    raw_cid = int(client_proxy.cid)
                except Exception:
                    raw_cid = len(client_eval_entries)
            cid = int(raw_cid)
            loss = float(eval_res.loss) if eval_res.loss is not None else 0.0
            num_samples = int(eval_res.num_examples)
            acc = float(m.get("accuracy", 0.0))

            entry = {
                "round": server_round,
                "client_id": cid,
                "test_loss": round(loss, 6),
                "test_accuracy": round(acc, 6),
                "num_samples": num_samples,
            }
            self.client_eval_records.append(entry)
            client_eval_entries.append(entry)
            client_accuracies.append(acc)
            client_losses.append(loss)

        mcta = float(np.mean(client_accuracies)) if client_accuracies else 0.0
        wcta = float(metrics_aggregated.get("accuracy", 0.0))
        mctl = float(np.mean(client_losses)) if client_losses else 0.0

        self.mcta_history[server_round] = mcta
        self.wcta_history[server_round] = wcta
        self.mctl_history[server_round] = mctl

        if mcta > self.bmcta:
            self.bmcta = mcta
            self.bmcta_round = server_round
            self.bmcta_loss = mctl

        if self.bta == 0.0 or wcta > self.bta:
            self.bta = wcta
            self.bta_round = server_round

        for r in self.round_records:
            if r["round"] == server_round:
                r["mcta"] = round(mcta, 6)
                r["mctl"] = round(mctl, 6)
                r["bmcta"] = round(self.bmcta, 6)
                r["bmcta_loss"] = round(self.bmcta_loss, 6) if self.bmcta_loss is not None else None
                r["wcta"] = round(wcta, 6)
                r["bta"] = round(self.bta, 6)
                break

    def record_parameter_dynamics(
        self,
        server_round: int,
        previous_parameters: Any,
        aggregated_parameters: Any,
        results: List[Tuple[Any, Any]],
    ):
        """Compute and record global model parameter updates, layer norms, and client drift/divergence."""
        agg_ndarrays = _to_ndarrays(aggregated_parameters)
        if agg_ndarrays is None or len(agg_ndarrays) == 0:
            return

        prev_ndarrays = _to_ndarrays(previous_parameters)
        has_prev = prev_ndarrays is not None and len(prev_ndarrays) == len(agg_ndarrays)

        # 1. Global weight vector L2 norm
        global_l2 = float(np.sqrt(sum(np.sum(np.square(w.astype(np.float64))) for w in agg_ndarrays)))

        # 2. Global parameter update delta (step magnitude)
        if has_prev:
            delta_l2 = float(np.sqrt(sum(np.sum(np.square((w_agg - w_prev).astype(np.float64))) for w_agg, w_prev in zip(agg_ndarrays, prev_ndarrays))))
            prev_l2 = float(np.sqrt(sum(np.sum(np.square(w.astype(np.float64))) for w in prev_ndarrays)))
            delta_rel = float(delta_l2 / max(prev_l2, 1e-8))
        else:
            delta_l2 = 0.0
            delta_rel = 0.0

        # 3. Layer-wise norms and deltas
        for idx, w_agg in enumerate(agg_ndarrays):
            l_norm = float(np.linalg.norm(w_agg.astype(np.float64)))
            l_delta = float(np.linalg.norm((w_agg - prev_ndarrays[idx]).astype(np.float64))) if has_prev else 0.0
            self.parameter_layer_records.append({
                "round": server_round,
                "layer_idx": idx,
                "layer_l2_norm": round(l_norm, 6),
                "layer_delta_l2": round(l_delta, 6),
            })

        # 4. Client-wise drift from previous global model & divergence from consensus
        client_drifts = []
        client_divergences = []
        for client_proxy, fit_res in results:
            m = getattr(fit_res, "metrics", {}) or {}
            raw_cid = m.get("client_id")
            if raw_cid is None:
                try:
                    raw_cid = int(client_proxy.cid)
                except Exception:
                    raw_cid = len(client_drifts)
            cid = int(raw_cid)
            device_type = str(m.get("device_type") or get_device_type(cid))

            c_ndarrays = _to_ndarrays(getattr(fit_res, "parameters", None))
            if c_ndarrays is None or len(c_ndarrays) != len(agg_ndarrays):
                continue

            c_norm = float(np.sqrt(sum(np.sum(np.square(w.astype(np.float64))) for w in c_ndarrays)))

            # Drift from previous global model
            if has_prev:
                drift_l2 = float(np.sqrt(sum(np.sum(np.square((w_c - w_prev).astype(np.float64))) for w_c, w_prev in zip(c_ndarrays, prev_ndarrays))))
                rel_drift = float(drift_l2 / max(prev_l2, 1e-8))
            else:
                drift_l2 = 0.0
                rel_drift = 0.0

            # Divergence from newly aggregated consensus
            divergence_l2 = float(np.sqrt(sum(np.sum(np.square((w_c - w_agg).astype(np.float64))) for w_c, w_agg in zip(c_ndarrays, agg_ndarrays))))

            client_drifts.append(drift_l2)
            client_divergences.append(divergence_l2)

            self.parameter_client_records.append({
                "round": server_round,
                "client_id": cid,
                "device_type": device_type,
                "client_weight_norm": round(c_norm, 6),
                "drift_from_global_l2": round(drift_l2, 6),
                "divergence_from_consensus_l2": round(divergence_l2, 6),
                "relative_drift": round(rel_drift, 6),
            })

        avg_drift = float(np.mean(client_drifts)) if client_drifts else 0.0
        max_drift = float(np.max(client_drifts)) if client_drifts else 0.0
        min_drift = float(np.min(client_drifts)) if client_drifts else 0.0
        avg_div = float(np.mean(client_divergences)) if client_divergences else 0.0
        max_div = float(np.max(client_divergences)) if client_divergences else 0.0

        round_entry = {
            "round": server_round,
            "global_l2_norm": round(global_l2, 6),
            "global_delta_l2": round(delta_l2, 6),
            "global_delta_relative": round(delta_rel, 6),
            "avg_client_drift_l2": round(avg_drift, 6),
            "max_client_drift_l2": round(max_drift, 6),
            "min_client_drift_l2": round(min_drift, 6),
            "avg_client_divergence_l2": round(avg_div, 6),
            "max_client_divergence_l2": round(max_div, 6),
            "num_clients": len(results),
        }
        self.parameter_round_records.append(round_entry)

    # ───────────────────────────────────────────────────────────
    # Summary Helper
    # ───────────────────────────────────────────────────────────

    def get_summary(self) -> Dict[str, Any]:
        """Return executive summary metrics for terminal and strategy returns."""
        total_time = time.time() - self.fl_start_time
        round_times = [r["round_time_seconds"] for r in self.round_records]
        avg_round = float(np.mean(round_times)) if round_times else 0.0

        best_metric = (
            self.best_eval_accuracy
            if self.best_eval_accuracy > 0
            else (
                self.round_records[-1]["train_accuracy_avg"]
                if self.round_records
                else 0.0
            )
        )

        final_eval_loss = (
            self.round_records[-1].get("eval_loss")
            if self.round_records and self.round_records[-1].get("eval_loss") is not None
            else (
                self.round_records[-1].get("mctl")
                if self.round_records and self.round_records[-1].get("mctl") is not None
                else None
            )
        )
        best_eval_loss = (
            self.best_eval_loss
            if self.best_eval_loss != float("inf")
            else (
                self.bmcta_loss
                if self.bmcta_loss is not None
                else None
            )
        )
        final_eval_acc = (
            self.round_records[-1].get("eval_accuracy")
            if self.round_records and self.round_records[-1].get("eval_accuracy") is not None
            else (
                self.round_records[-1].get("mcta")
                if self.round_records and self.round_records[-1].get("mcta") is not None
                else None
            )
        )
        best_eval_acc = (
            self.best_eval_accuracy
            if self.best_eval_accuracy > 0
            else (
                self.bmcta
                if self.bmcta > 0
                else None
            )
        )

        summary = {
            "strategy": self.strategy_name,
            "model_name": self.model_name,
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "weight_payload_size_kb": round(self.model_weight_size_kb, 2)
            if self.model_weight_size_kb is not None
            else None,
            "checkpoint_file_size_kb": round(self.model_file_size_kb, 2)
            if self.model_file_size_kb is not None
            else None,
            "total_rounds": len(self.round_records),
            "total_time_seconds": round(total_time, 2),
            "avg_round_time_seconds": round(avg_round, 2),
            "best_metric": round(best_metric, 4),
            "best_round": self.best_eval_round,
            "best_eval_accuracy": round(best_eval_acc, 4) if best_eval_acc is not None else None,
            "final_eval_accuracy": round(final_eval_acc, 4) if final_eval_acc is not None else None,
            "best_eval_loss": round(best_eval_loss, 4) if best_eval_loss is not None else None,
            "final_eval_loss": round(final_eval_loss, 4) if final_eval_loss is not None else None,
            "bmcta": round(self.bmcta, 4) if self.bmcta > 0 else None,
            "bmcta_round": self.bmcta_round if self.bmcta > 0 else None,
            "bmcta_loss": round(self.bmcta_loss, 4) if self.bmcta_loss is not None else None,
            "bta": round(self.bta, 4) if self.bta > 0 else None,
            "bta_round": self.bta_round if self.bta > 0 else None,
            "local_test_enabled": self.local_test_enabled,
        }
        if self.proximal_mu is not None:
            summary["proximal_mu"] = self.proximal_mu
        return summary

    # ───────────────────────────────────────────────────────────
    # CSV Exports
    # ───────────────────────────────────────────────────────────

    def export_csvs(self, save_dir: str) -> Dict[str, str]:
        """Export all client, epoch, and round metrics into CSV files."""
        os.makedirs(save_dir, exist_ok=True)
        paths = {}

        # 1. client_round_metrics.csv
        client_csv_path = os.path.join(save_dir, "client_round_metrics.csv")
        client_columns = [
            "round",
            "client_id",
            "device_type",
            "num_samples",
            "local_epochs",
            "learning_rate",
            "training_time_seconds",
            "epoch_time_avg",
            "epoch_time_min",
            "epoch_time_max",
            "train_loss",
            "train_accuracy",
            "ping_ms",
            "weight_size_kb",
            "cpu_percent",
            "ram_percent",
            "ram_used_mb",
            "gpu_memory_mb",
        ]
        if self.client_records:
            if pd is not None:
                df_clients = pd.DataFrame(
                    [
                        {k: rec[k] for k in client_columns if k in rec}
                        for rec in self.client_records
                    ]
                )
                df_clients.to_csv(client_csv_path, index=False)
            else:
                import csv
                with open(client_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=client_columns)
                    writer.writeheader()
                    for rec in self.client_records:
                        writer.writerow({k: rec.get(k, "") for k in client_columns})
        else:
            with open(client_csv_path, "w", encoding="utf-8") as f:
                f.write(",".join(client_columns) + "\n")
        paths["client_round_metrics_csv"] = client_csv_path

        # 2. client_epoch_metrics.csv
        epoch_csv_path = os.path.join(save_dir, "client_epoch_metrics.csv")
        epoch_columns = [
            "round",
            "client_id",
            "device_type",
            "epoch",
            "epoch_time_seconds",
            "cumulative_epoch_time_seconds",
            "train_loss",
            "train_accuracy",
            "learning_rate",
        ]
        if self.epoch_records:
            if pd is not None:
                df_epochs = pd.DataFrame(self.epoch_records)
                df_epochs.to_csv(epoch_csv_path, index=False)
            else:
                import csv
                with open(epoch_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=epoch_columns)
                    writer.writeheader()
                    writer.writerows(self.epoch_records)
        else:
            with open(epoch_csv_path, "w", encoding="utf-8") as f:
                f.write(",".join(epoch_columns) + "\n")
        paths["client_epoch_metrics_csv"] = epoch_csv_path

        # 3. round_metrics.csv
        round_csv_path = os.path.join(save_dir, "round_metrics.csv")
        round_columns = [
            "round",
            "round_time_seconds",
            "total_elapsed_seconds",
            "num_clients_reporting",
            "train_loss_avg",
            "train_accuracy_avg",
            "eval_loss",
            "eval_accuracy",
            "mcta",
            "mctl",
            "bmcta",
            "bmcta_round",
            "bmcta_loss",
            "wcta",
            "bta",
            "client_train_time_avg",
            "client_train_time_min",
            "client_train_time_max",
            "straggler_time_seconds",
            "epoch_time_avg",
            "total_weight_size_kb",
            "server_overhead_seconds",
            "ping_ms_avg",
            "ping_ms_min",
            "ping_ms_max",
        ]
        if self.round_records:
            if pd is not None:
                df_rounds = pd.DataFrame(self.round_records)
                df_rounds.to_csv(round_csv_path, index=False)
            else:
                import csv
                with open(round_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=round_columns)
                    writer.writeheader()
                    writer.writerows(self.round_records)
        else:
            with open(round_csv_path, "w", encoding="utf-8") as f:
                f.write(",".join(round_columns) + "\n")
        paths["round_metrics_csv"] = round_csv_path

        # 3b. client_eval_metrics.csv (Federated client evaluation records)
        client_eval_csv_path = os.path.join(save_dir, "client_eval_metrics.csv")
        client_eval_columns = [
            "round",
            "client_id",
            "test_loss",
            "test_accuracy",
            "num_samples",
        ]
        if self.client_eval_records:
            if pd is not None:
                df_eval = pd.DataFrame(self.client_eval_records)
                df_eval.to_csv(client_eval_csv_path, index=False)
            else:
                import csv
                with open(client_eval_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=client_eval_columns)
                    writer.writeheader()
                    writer.writerows(self.client_eval_records)
            paths["client_eval_metrics_csv"] = client_eval_csv_path

        # 4. parameters_changes.csv (Global round-level parameter changes)
        param_csv_path = os.path.join(save_dir, "parameters_changes.csv")
        param_columns = [
            "round",
            "global_l2_norm",
            "global_delta_l2",
            "global_delta_relative",
            "avg_client_drift_l2",
            "max_client_drift_l2",
            "min_client_drift_l2",
            "avg_client_divergence_l2",
            "max_client_divergence_l2",
            "num_clients",
        ]
        if self.parameter_round_records:
            if pd is not None:
                pd.DataFrame(self.parameter_round_records).to_csv(param_csv_path, index=False)
            else:
                import csv
                with open(param_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=param_columns)
                    writer.writeheader()
                    writer.writerows(self.parameter_round_records)
            paths["parameters_changes_csv"] = param_csv_path

        # 5. client_parameters_drift.csv (Per-client drift records)
        client_drift_csv_path = os.path.join(save_dir, "client_parameters_drift.csv")
        if self.parameter_client_records:
            if pd is not None:
                pd.DataFrame(self.parameter_client_records).to_csv(client_drift_csv_path, index=False)
            else:
                import csv
                c_drift_cols = ["round", "client_id", "device_type", "client_weight_norm", "drift_from_global_l2", "divergence_from_consensus_l2", "relative_drift"]
                with open(client_drift_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=c_drift_cols)
                    writer.writeheader()
                    writer.writerows(self.parameter_client_records)
            paths["client_parameters_drift_csv"] = client_drift_csv_path

        # 6. layer_parameters_changes.csv (Layer-wise norms & deltas)
        layer_csv_path = os.path.join(save_dir, "layer_parameters_changes.csv")
        if self.parameter_layer_records:
            if pd is not None:
                pd.DataFrame(self.parameter_layer_records).to_csv(layer_csv_path, index=False)
            paths["layer_parameters_changes_csv"] = layer_csv_path

        return paths

    # ───────────────────────────────────────────────────────────
    # Enhanced JSON Export
    # ───────────────────────────────────────────────────────────

    def export_json(
        self,
        save_dir: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
        plot_files: Optional[List[str]] = None,
    ) -> str:
        """Generate comprehensive, deep fl_results.json."""
        os.makedirs(save_dir, exist_ok=True)
        total_time = time.time() - self.fl_start_time
        round_times = [r["round_time_seconds"] for r in self.round_records]
        client_times = [c["training_time_seconds"] for c in self.client_records]
        straggler_times = [r["straggler_time_seconds"] for r in self.round_records]

        # Calculate client summaries
        client_ids = sorted(list(set(c["client_id"] for c in self.client_records)))
        client_summaries = {}
        for cid in client_ids:
            c_recs = [c for c in self.client_records if c["client_id"] == cid]
            c_ep_recs = [e for e in self.epoch_records if e["client_id"] == cid]

            c_r_times = [c["training_time_seconds"] for c in c_recs]
            c_ep_times = [e["epoch_time_seconds"] for e in c_ep_recs]
            c_cpu = [c["cpu_percent"] for c in c_recs if c["cpu_percent"] > 0]
            c_ram = [c["ram_percent"] for c in c_recs if c["ram_percent"] > 0]
            c_gpu = [c["gpu_memory_mb"] for c in c_recs if c["gpu_memory_mb"] > 0]
            c_pings = [c["ping_ms"] for c in c_recs if c.get("ping_ms") is not None]

            client_summaries[str(cid)] = {
                "client_id": cid,
                "device_type": c_recs[0]["device_type"] if c_recs else "pc",
                "num_samples": c_recs[0]["num_samples"] if c_recs else 0,
                "total_rounds_participated": len(c_recs),
                "total_training_time_seconds": round(float(sum(c_r_times)), 4),
                "avg_time_per_round_seconds": round(float(np.mean(c_r_times)), 4)
                if c_r_times
                else 0.0,
                "min_time_per_round_seconds": round(float(min(c_r_times)), 4)
                if c_r_times
                else 0.0,
                "max_time_per_round_seconds": round(float(max(c_r_times)), 4)
                if c_r_times
                else 0.0,
                "avg_time_per_epoch_seconds": round(float(np.mean(c_ep_times)), 4)
                if c_ep_times
                else 0.0,
                "min_time_per_epoch_seconds": round(float(min(c_ep_times)), 4)
                if c_ep_times
                else 0.0,
                "max_time_per_epoch_seconds": round(float(max(c_ep_times)), 4)
                if c_ep_times
                else 0.0,
                "total_epochs_trained": len(c_ep_recs),
                "final_train_loss": c_recs[-1]["train_loss"] if c_recs else None,
                "final_train_accuracy": c_recs[-1]["train_accuracy"]
                if c_recs
                else None,
                "total_data_uploaded_kb": round(
                    float(sum(c["weight_size_kb"] for c in c_recs)), 2
                ),
                "avg_cpu_percent": round(float(np.mean(c_cpu)), 2) if c_cpu else 0.0,
                "avg_ram_percent": round(float(np.mean(c_ram)), 2) if c_ram else 0.0,
                "avg_gpu_memory_mb": round(float(np.mean(c_gpu)), 2) if c_gpu else 0.0,
                "avg_ping_ms": round(float(np.mean(c_pings)), 2) if c_pings else None,
                "min_ping_ms": round(float(min(c_pings)), 2) if c_pings else None,
                "max_ping_ms": round(float(max(c_pings)), 2) if c_pings else None,
            }

        # Hardware heterogeneity breakdown
        device_types = sorted(
            list(set(c["device_type"] for c in self.client_records))
        )
        heterogeneity_stats = {
            "device_types_present": device_types,
            "by_device_type": {},
        }
        for dtype in device_types:
            dtype_c_recs = [
                c for c in self.client_records if c["device_type"] == dtype
            ]
            dtype_ep_recs = [
                e for e in self.epoch_records if e["device_type"] == dtype
            ]
            unique_clients = len(set(c["client_id"] for c in dtype_c_recs))
            r_times = [c["training_time_seconds"] for c in dtype_c_recs]
            ep_times = [e["epoch_time_seconds"] for e in dtype_ep_recs]
            dtype_pings = [c["ping_ms"] for c in dtype_c_recs if c.get("ping_ms") is not None]

            heterogeneity_stats["by_device_type"][dtype] = {
                "client_count": unique_clients,
                "avg_round_training_time_seconds": round(float(np.mean(r_times)), 4)
                if r_times
                else 0.0,
                "avg_epoch_time_seconds": round(float(np.mean(ep_times)), 4)
                if ep_times
                else 0.0,
                "avg_ping_ms": round(float(np.mean(dtype_pings)), 2) if dtype_pings else None,
                "total_device_compute_time_seconds": round(float(sum(r_times)), 4),
            }

        # Calculate speedup ratio if multiple device types exist
        if len(device_types) > 1:
            avg_ep_by_type = [
                heterogeneity_stats["by_device_type"][d]["avg_epoch_time_seconds"]
                for d in device_types
                if heterogeneity_stats["by_device_type"][d]["avg_epoch_time_seconds"]
                > 0
            ]
            if avg_ep_by_type and min(avg_ep_by_type) > 0:
                heterogeneity_stats["speedup_ratio_fastest_vs_slowest"] = round(
                    max(avg_ep_by_type) / min(avg_ep_by_type), 2
                )

        # Assemble full history combining round record with client snapshots
        rounds_history = []
        for r_entry in self.round_records:
            rnd = r_entry["round"]
            c_snapshots = [
                {
                    "client_id": c["client_id"],
                    "device_type": c["device_type"],
                    "num_samples": c["num_samples"],
                    "training_time_seconds": c["training_time_seconds"],
                    "epoch_time_avg": c["epoch_time_avg"],
                    "epoch_times": c["epoch_times"],
                    "train_loss": c["train_loss"],
                    "train_accuracy": c["train_accuracy"],
                    "ping_ms": c.get("ping_ms"),
                    "epoch_losses": c["epoch_losses"],
                    "epoch_accuracies": c["epoch_accuracies"],
                    "weight_size_kb": c["weight_size_kb"],
                    "cpu_percent": c["cpu_percent"],
                    "ram_percent": c["ram_percent"],
                    "gpu_memory_mb": c["gpu_memory_mb"],
                }
                for c in self.client_records
                if c["round"] == rnd
            ]
            entry_copy = dict(r_entry)
            entry_copy["clients"] = c_snapshots
            rounds_history.append(entry_copy)

        # Final accuracy/loss determinations
        final_eval_acc = (
            self.round_records[-1]["eval_accuracy"]
            if self.round_records and self.round_records[-1]["eval_accuracy"] is not None
            else None
        )
        final_eval_loss = (
            self.round_records[-1]["eval_loss"]
            if self.round_records and self.round_records[-1]["eval_loss"] is not None
            else None
        )
        final_client_test_acc = (
            self.round_records[-1].get("mcta")
            if self.round_records and self.round_records[-1].get("mcta") is not None
            else None
        )
        final_client_test_loss = (
            self.round_records[-1].get("mctl")
            if self.round_records and self.round_records[-1].get("mctl") is not None
            else None
        )

        best_eval_acc_val = (
            round(self.best_eval_accuracy, 4)
            if self.best_eval_accuracy > 0
            else (round(self.bmcta, 4) if self.bmcta > 0 else None)
        )
        best_eval_loss_val = (
            round(self.best_eval_loss, 4)
            if self.best_eval_loss != float("inf")
            else (round(self.bmcta_loss, 4) if self.bmcta_loss is not None else None)
        )
        final_eval_acc_resolved = final_eval_acc if final_eval_acc is not None else final_client_test_acc
        final_eval_loss_resolved = final_eval_loss if final_eval_loss is not None else final_client_test_loss

        final_train_acc = (
            self.round_records[-1]["train_accuracy_avg"]
            if self.round_records
            else None
        )
        final_train_loss = (
            self.round_records[-1]["train_loss_avg"] if self.round_records else None
        )

        total_comm_kb = float(
            sum(r["total_weight_size_kb"] for r in self.round_records)
        )

        total_straggler = float(sum(straggler_times))
        total_wall_time = (
            float(self.round_records[-1]["total_elapsed_seconds"])
            if self.round_records
            else total_time
        )
        straggler_overhead_pct = (
            round((total_straggler / total_wall_time) * 100, 2)
            if total_wall_time > 0
            else 0.0
        )

        meta = {
            "strategy": self.strategy_name,
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "status": "completed",
            "num_rounds_completed": len(self.round_records),
        }
        if extra_metadata:
            meta.update(extra_metadata)

        full_results = {
            "metadata": meta,
            "summary": {
                "total_wall_clock_time_seconds": round(total_wall_time, 2),
                "total_client_compute_time_seconds": round(float(sum(client_times)), 2),
                "avg_round_time_seconds": round(float(np.mean(round_times)), 2)
                if round_times
                else 0.0,
                "min_round_time_seconds": round(float(min(round_times)), 2)
                if round_times
                else 0.0,
                "max_round_time_seconds": round(float(max(round_times)), 2)
                if round_times
                else 0.0,
                "total_straggler_overhead_seconds": round(total_straggler, 2),
                "straggler_overhead_percent": straggler_overhead_pct,
                "final_global_accuracy": final_eval_acc,
                "best_global_accuracy": round(self.best_eval_accuracy, 4)
                if self.best_eval_accuracy > 0
                else None,
                "best_global_accuracy_round": self.best_eval_round
                if self.best_eval_accuracy > 0
                else None,
                "final_global_loss": final_eval_loss,
                "best_global_loss": round(self.best_eval_loss, 4)
                if self.best_eval_loss != float("inf")
                else None,
                "final_eval_accuracy": round(final_eval_acc_resolved, 4) if final_eval_acc_resolved is not None else None,
                "best_eval_accuracy": best_eval_acc_val,
                "final_eval_loss": round(final_eval_loss_resolved, 4) if final_eval_loss_resolved is not None else None,
                "best_eval_loss": best_eval_loss_val,
                "final_client_test_accuracy": round(final_client_test_acc, 4) if final_client_test_acc is not None else None,
                "final_client_test_loss": round(final_client_test_loss, 4) if final_client_test_loss is not None else None,
                "final_train_loss_avg": final_train_loss,
                "final_train_accuracy_avg": final_train_acc,
                "total_communication_kb": round(total_comm_kb, 2),
                "total_communication_mb": round(total_comm_kb / 1024.0, 2),
                "model_name": self.model_name,
                "total_parameters": self.total_parameters,
                "trainable_parameters": self.trainable_parameters,
                "weight_payload_size_kb": round(self.model_weight_size_kb, 2)
                if self.model_weight_size_kb is not None
                else None,
                "checkpoint_file_size_kb": round(self.model_file_size_kb, 2)
                if self.model_file_size_kb is not None
                else None,
                "bmcta": round(self.bmcta, 4) if self.bmcta > 0 else None,
                "bmcta_round": self.bmcta_round if self.bmcta > 0 else None,
                "bmcta_loss": round(self.bmcta_loss, 4) if self.bmcta_loss is not None else None,
                "bta": round(self.bta, 4) if self.bta > 0 else None,
                "bta_round": self.bta_round if self.bta > 0 else None,
                "local_test_enabled": self.local_test_enabled,
            },
            "model_info": {
                "model_name": self.model_name,
                "total_parameters": self.total_parameters,
                "trainable_parameters": self.trainable_parameters,
                "weight_payload_size_kb": round(self.model_weight_size_kb, 2)
                if self.model_weight_size_kb is not None
                else None,
                "weight_payload_size_mb": round(self.model_weight_size_kb / 1024.0, 3)
                if self.model_weight_size_kb is not None
                else None,
                "checkpoint_file_size_kb": round(self.model_file_size_kb, 2)
                if self.model_file_size_kb is not None
                else None,
                "checkpoint_file_size_mb": round(self.model_file_size_kb / 1024.0, 3)
                if self.model_file_size_kb is not None
                else None,
            },
            "paper_benchmark_metrics": {
                "local_test_enabled": self.local_test_enabled,
                "bmcta": round(self.bmcta, 4) if self.bmcta > 0 else None,
                "bmcta_round": self.bmcta_round if self.bmcta > 0 else None,
                "bmcta_loss": round(self.bmcta_loss, 4) if self.bmcta_loss is not None else None,
                "bta": round(self.bta, 4) if self.bta > 0 else None,
                "bta_round": self.bta_round if self.bta > 0 else None,
                "mcta_trajectory": [
                    {
                        "round": r,
                        "mcta": round(self.mcta_history[r], 4),
                        "mctl": round(self.mctl_history[r], 4) if r in self.mctl_history else None,
                        "wcta": round(self.wcta_history[r], 4),
                    }
                    for r in sorted(self.mcta_history.keys())
                ],
            },
            "client_summaries": client_summaries,
            "hardware_heterogeneity": heterogeneity_stats,
            "rounds_history": rounds_history,
            "artifacts": {
                "client_round_metrics_csv": "client_round_metrics.csv",
                "client_epoch_metrics_csv": "client_epoch_metrics.csv",
                "round_metrics_csv": "round_metrics.csv",
                "client_eval_metrics_csv": "client_eval_metrics.csv",
                "plots": [os.path.basename(p) for p in (plot_files or [])],
            },
        }

        if self.parameter_round_records:
            full_results["parameter_dynamics"] = {
                "summary": {
                    "initial_global_l2_norm": self.parameter_round_records[0]["global_l2_norm"],
                    "final_global_l2_norm": self.parameter_round_records[-1]["global_l2_norm"],
                    "total_parameter_update_distance": round(float(sum(r["global_delta_l2"] for r in self.parameter_round_records)), 6),
                    "avg_client_drift_l2": round(float(np.mean([r["avg_client_drift_l2"] for r in self.parameter_round_records])), 6),
                    "max_client_drift_l2": round(float(max(r["max_client_drift_l2"] for r in self.parameter_round_records)), 6),
                    "avg_client_divergence_l2": round(float(np.mean([r["avg_client_divergence_l2"] for r in self.parameter_round_records])), 6),
                },
                "rounds": self.parameter_round_records,
            }
            param_json_path = os.path.join(save_dir, "parameters_records.json")
            param_payload = {
                "strategy": self.strategy_name,
                "total_rounds": len(self.parameter_round_records),
                "round_dynamics": self.parameter_round_records,
                "client_drift": self.parameter_client_records,
                "layer_dynamics": self.parameter_layer_records,
            }
            with open(param_json_path, "w", encoding="utf-8") as pf:
                json.dump(param_payload, pf, indent=4, ensure_ascii=False)
            full_results["artifacts"]["parameters_records_json"] = "parameters_records.json"
            full_results["artifacts"]["parameters_changes_csv"] = "parameters_changes.csv"
            full_results["artifacts"]["client_parameters_drift_csv"] = "client_parameters_drift.csv"

        json_path = os.path.join(save_dir, "fl_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=4, ensure_ascii=False, default=str)
        return json_path

    # ───────────────────────────────────────────────────────────
    # Comparison Plots
    # ───────────────────────────────────────────────────────────

    def generate_plots(self, save_dir: str) -> List[str]:
        """Generate 5 rich comparison and diagnostic charts."""
        os.makedirs(save_dir, exist_ok=True)
        if not self.round_records or not self.client_records:
            return []

        generated_plots = []
        client_ids = sorted(list(set(c["client_id"] for c in self.client_records)))
        palette = plt.cm.tab10(np.linspace(0, 1, max(10, len(client_ids))))
        client_color_map = {cid: palette[i % 10] for i, cid in enumerate(client_ids)}

        # -------------------------------------------------------------
        # 1. client_training_time_comparison.png
        # -------------------------------------------------------------
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 11))

            # (A) Client Training Time per Round
            ax = axes[0, 0]
            for cid in client_ids:
                c_recs = [c for c in self.client_records if c["client_id"] == cid]
                c_recs.sort(key=lambda x: x["round"])
                r_nums = [c["round"] for c in c_recs]
                t_vals = [c["training_time_seconds"] for c in c_recs]
                dtype = c_recs[0]["device_type"]
                ax.plot(
                    r_nums,
                    t_vals,
                    marker="o",
                    linewidth=2,
                    label=f"Client {cid} ({dtype})",
                    color=client_color_map[cid],
                )
            ax.set_title("Client Training Time per Round", fontsize=13, fontweight="bold")
            ax.set_xlabel("FL Round", fontsize=11)
            ax.set_ylabel("Training Time (seconds)", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(loc="best", fontsize=9)

            # (B) Average Training Time per Client (Bar Chart with Std)
            ax = axes[0, 1]
            c_labels = []
            c_means = []
            c_stds = []
            bar_colors = []
            for cid in client_ids:
                c_recs = [c for c in self.client_records if c["client_id"] == cid]
                times = [c["training_time_seconds"] for c in c_recs]
                dtype = c_recs[0]["device_type"]
                c_labels.append(f"C{cid}\n({dtype})")
                c_means.append(float(np.mean(times)))
                c_stds.append(float(np.std(times)))
                bar_colors.append(client_color_map[cid])

            bars = ax.bar(
                range(len(client_ids)),
                c_means,
                yerr=c_stds,
                capsize=5,
                color=bar_colors,
                alpha=0.85,
                edgecolor="black",
            )
            ax.set_xticks(range(len(client_ids)))
            ax.set_xticklabels(c_labels, fontsize=9)
            ax.set_title("Average Training Time per Round by Client", fontsize=13, fontweight="bold")
            ax.set_ylabel("Time (seconds)", fontsize=11)
            ax.grid(axis="y", linestyle="--", alpha=0.6)
            for bar in bars:
                h = bar.get_height()
                ax.annotate(
                    f"{h:.1f}s",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

            # (C) Straggler Latency Penalty per Round
            ax = axes[1, 0]
            rounds = [r["round"] for r in self.round_records]
            straggler_delays = [r["straggler_time_seconds"] for r in self.round_records]
            ax.fill_between(rounds, straggler_delays, color="coral", alpha=0.3)
            ax.plot(rounds, straggler_delays, color="crimson", marker="s", linewidth=2)
            ax.set_title(
                "Straggler Latency Penalty per Round (Max - Min Client Time)",
                fontsize=13,
                fontweight="bold",
            )
            ax.set_xlabel("FL Round", fontsize=11)
            ax.set_ylabel("Wait Delay (seconds)", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.6)

            # (D) Total Cumulative Training Time Comparison
            ax = axes[1, 1]
            cum_times = []
            for cid in client_ids:
                c_recs = [c for c in self.client_records if c["client_id"] == cid]
                c_recs.sort(key=lambda x: x["round"])
                t_vals = [c["training_time_seconds"] for c in c_recs]
                cum_times.append(sum(t_vals))

            ax.barh(
                range(len(client_ids)),
                cum_times,
                color=bar_colors,
                alpha=0.85,
                edgecolor="black",
            )
            ax.set_yticks(range(len(client_ids)))
            ax.set_yticklabels(c_labels, fontsize=9)
            ax.set_title("Total Cumulative Active Training Time", fontsize=13, fontweight="bold")
            ax.set_xlabel("Cumulative Compute Time (seconds)", fontsize=11)
            ax.grid(axis="x", linestyle="--", alpha=0.6)
            for i, val in enumerate(cum_times):
                ax.text(val + max(cum_times) * 0.02, i, f"{val:.1f}s", va="center", fontsize=9, fontweight="bold")

            plt.suptitle(
                f"FedMedAI [{self.strategy_name.upper()}] — Client Training Time & Straggler Analysis",
                fontsize=15,
                fontweight="bold",
                y=0.995,
            )
            plt.tight_layout()
            out_p = os.path.join(save_dir, "client_training_time_comparison.png")
            plt.savefig(out_p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            generated_plots.append(out_p)
        except Exception as e:
            print(f"[MetricsRecorder] Error plotting training time comparison: {e}")

        # -------------------------------------------------------------
        # 2. client_epoch_time_comparison.png
        # -------------------------------------------------------------
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            # (A) Boxplot / distribution of epoch times per client
            ax = axes[0]
            epoch_data_by_client = []
            for cid in client_ids:
                times = [
                    e["epoch_time_seconds"]
                    for e in self.epoch_records
                    if e["client_id"] == cid
                ]
                epoch_data_by_client.append(times if times else [0.0])

            boxplot_kwargs = {"patch_artist": True, "medianprops": dict(color="black", linewidth=1.5)}
            try:
                bplot = ax.boxplot(
                    epoch_data_by_client,
                    tick_labels=[f"C{cid}" for cid in client_ids],
                    **boxplot_kwargs,
                )
            except TypeError:
                bplot = ax.boxplot(
                    epoch_data_by_client,
                    labels=[f"C{cid}" for cid in client_ids],
                    **boxplot_kwargs,
                )
            for patch, cid in zip(bplot["boxes"], client_ids):
                patch.set_facecolor(client_color_map[cid])
                patch.set_alpha(0.7)

            ax.set_title("Epoch Duration Distribution per Client", fontsize=13, fontweight="bold")
            ax.set_xlabel("Client ID", fontsize=11)
            ax.set_ylabel("Time per Epoch (seconds)", fontsize=11)
            ax.grid(axis="y", linestyle="--", alpha=0.6)

            # (B) Average Epoch Time Grouped by Device Type
            ax = axes[1]
            device_types = sorted(list(set(c["device_type"] for c in self.client_records)))
            dev_means = []
            dev_stds = []
            base_palette = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#1b9e77", "#e6ab02", "#a6761d"]
            dev_colors = [base_palette[i % len(base_palette)] for i in range(len(device_types))]
            for d in device_types:
                d_ep_times = [
                    e["epoch_time_seconds"]
                    for e in self.epoch_records
                    if e["device_type"] == d
                ]
                dev_means.append(float(np.mean(d_ep_times)) if d_ep_times else 0.0)
                dev_stds.append(float(np.std(d_ep_times)) if d_ep_times else 0.0)

            dev_bars = ax.bar(
                device_types,
                dev_means,
                yerr=dev_stds,
                capsize=6,
                color=dev_colors,
                alpha=0.85,
                edgecolor="black",
            )
            ax.set_title("Mean Epoch Duration by Device Class", fontsize=13, fontweight="bold")
            ax.set_ylabel("Seconds per Epoch", fontsize=11)
            ax.grid(axis="y", linestyle="--", alpha=0.6)
            for bar in dev_bars:
                h = bar.get_height()
                ax.annotate(
                    f"{h:.2f}s",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

            plt.suptitle(
                f"FedMedAI [{self.strategy_name.upper()}] — Per-Epoch Computation Time Analysis",
                fontsize=15,
                fontweight="bold",
            )
            plt.tight_layout()
            out_p = os.path.join(save_dir, "client_epoch_time_comparison.png")
            plt.savefig(out_p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            generated_plots.append(out_p)
        except Exception as e:
            print(f"[MetricsRecorder] Error plotting epoch time comparison: {e}")

        # -------------------------------------------------------------
        # 3. fl_training_curves.png
        # -------------------------------------------------------------
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
            rounds = [r["round"] for r in self.round_records]

            # (A) Loss over Rounds (Train vs Global Eval)
            ax = axes[0]
            train_losses = [r["train_loss_avg"] for r in self.round_records]
            ax.plot(rounds, train_losses, "b-o", linewidth=2, label="Client Train Loss (Avg)")
            eval_losses = [r["eval_loss"] for r in self.round_records]
            if any(l is not None for l in eval_losses):
                valid_r = [r for r, l in zip(rounds, eval_losses) if l is not None]
                valid_l = [l for l in eval_losses if l is not None]
                ax.plot(valid_r, valid_l, "r-s", linewidth=2, label="Global Eval Loss")
            ax.set_title("Loss over FL Rounds", fontsize=13, fontweight="bold")
            ax.set_xlabel("FL Round", fontsize=11)
            ax.set_ylabel("Loss", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(loc="best", fontsize=10)

            # (B) Accuracy over Rounds (Train vs Global Eval)
            ax = axes[1]
            train_accs = [r["train_accuracy_avg"] for r in self.round_records]
            ax.plot(rounds, train_accs, "b-o", linewidth=2, label="Client Train Acc (Avg)")
            eval_accs = [r["eval_accuracy"] for r in self.round_records]
            if any(a is not None for a in eval_accs):
                valid_r = [r for r, a in zip(rounds, eval_accs) if a is not None]
                valid_a = [a for a in eval_accs if a is not None]
                ax.plot(valid_r, valid_a, "g-^", linewidth=2, label="Global Eval Acc")
                if self.best_eval_accuracy > 0:
                    ax.axhline(
                        y=self.best_eval_accuracy,
                        color="darkgreen",
                        linestyle=":",
                        label=f"Best Eval: {self.best_eval_accuracy:.4f}",
                    )
            ax.set_title("Accuracy over FL Rounds", fontsize=13, fontweight="bold")
            ax.set_xlabel("FL Round", fontsize=11)
            ax.set_ylabel("Accuracy", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(loc="best", fontsize=10)

            # (C) Client-specific Training Accuracies (Dispersion)
            ax = axes[2]
            for cid in client_ids:
                c_recs = [c for c in self.client_records if c["client_id"] == cid]
                c_recs.sort(key=lambda x: x["round"])
                r_nums = [c["round"] for c in c_recs]
                accs = [c["train_accuracy"] for c in c_recs]
                ax.plot(
                    r_nums,
                    accs,
                    marker=".",
                    alpha=0.8,
                    label=f"Client {cid}",
                    color=client_color_map[cid],
                )
            ax.set_title("Client Local Accuracy Trajectories", fontsize=13, fontweight="bold")
            ax.set_xlabel("FL Round", fontsize=11)
            ax.set_ylabel("Local Train Accuracy", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(loc="best", fontsize=8)

            plt.suptitle(
                f"FedMedAI [{self.strategy_name.upper()}] — Convergence & Learning Curves",
                fontsize=15,
                fontweight="bold",
            )
            plt.tight_layout()
            out_p = os.path.join(save_dir, "fl_training_curves.png")
            plt.savefig(out_p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            generated_plots.append(out_p)
        except Exception as e:
            print(f"[MetricsRecorder] Error plotting learning curves: {e}")

        # -------------------------------------------------------------
        # 4. round_time_breakdown.png
        # -------------------------------------------------------------
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # (A) Stacked Latency Components per Round
            ax = axes[0]
            rounds = [r["round"] for r in self.round_records]
            c_mins = [r["client_train_time_min"] for r in self.round_records]
            stragglers = [r["straggler_time_seconds"] for r in self.round_records]
            overheads = [r["server_overhead_seconds"] for r in self.round_records]

            ax.bar(rounds, c_mins, label="Fastest Client Compute", color="#2b5c8f", alpha=0.85)
            ax.bar(
                rounds,
                stragglers,
                bottom=c_mins,
                label="Straggler Idle Wait (Max-Min)",
                color="#e6550d",
                alpha=0.85,
            )
            bottom_2 = [m + s for m, s in zip(c_mins, stragglers)]
            ax.bar(
                rounds,
                overheads,
                bottom=bottom_2,
                label="Server Comm & Aggregation",
                color="#756bb1",
                alpha=0.85,
            )
            ax.set_title("Round Duration Decomposition", fontsize=13, fontweight="bold")
            ax.set_xlabel("FL Round", fontsize=11)
            ax.set_ylabel("Duration (seconds)", fontsize=11)
            ax.grid(axis="y", linestyle="--", alpha=0.6)
            ax.legend(loc="upper left", fontsize=10)

            # (B) Cumulative Wall-clock vs Sum of Client Compute Time
            ax = axes[1]
            elapsed_wall = [r["total_elapsed_seconds"] for r in self.round_records]
            cum_client_compute = []
            cum_so_far = 0.0
            for r in self.round_records:
                rnd = r["round"]
                rnd_sum = sum(
                    c["training_time_seconds"]
                    for c in self.client_records
                    if c["round"] == rnd
                )
                cum_so_far += rnd_sum
                cum_client_compute.append(cum_so_far)

            ax.plot(
                rounds,
                elapsed_wall,
                "k-o",
                linewidth=2.5,
                label="Wall-clock Elapsed Time",
            )
            ax.plot(
                rounds,
                cum_client_compute,
                "b--s",
                linewidth=2,
                label="Aggregate Client Compute Time",
            )
            ax.set_title("Cumulative Time Progression", fontsize=13, fontweight="bold")
            ax.set_xlabel("FL Round", fontsize=11)
            ax.set_ylabel("Time (seconds)", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(loc="upper left", fontsize=10)

            plt.suptitle(
                f"FedMedAI [{self.strategy_name.upper()}] — Round Latency & System Efficiency Breakdown",
                fontsize=15,
                fontweight="bold",
            )
            plt.tight_layout()
            out_p = os.path.join(save_dir, "round_time_breakdown.png")
            plt.savefig(out_p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            generated_plots.append(out_p)
        except Exception as e:
            print(f"[MetricsRecorder] Error plotting round time breakdown: {e}")

        # -------------------------------------------------------------
        # 5. fl_summary_card.png (Visual Dashboard Card)
        # -------------------------------------------------------------
        try:
            fig, ax = plt.subplots(figsize=(10, 15.5))
            ax.axis("off")

            # Colors
            c_header = "#1b2838"
            c_sec_a = "#1f4e78"
            c_sec_b = "#2e75b6"
            c_row_light = "#f2f7fb"
            c_row_dark = "#ffffff"

            y = 0.985

            def draw_sec_header(ax, y_pos, title, col):
                ax.add_patch(
                    patches.Rectangle(
                        (0, y_pos - 0.026),
                        1,
                        0.030,
                        transform=ax.transAxes,
                        color=col,
                        zorder=2,
                    )
                )
                ax.text(
                    0.5,
                    y_pos - 0.010,
                    title,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11.5,
                    fontweight="bold",
                    color="white",
                    zorder=3,
                )
                return y_pos - 0.030

            def draw_entry(ax, y_pos, label, value, bg_col):
                ax.add_patch(
                    patches.Rectangle(
                        (0, y_pos - 0.022),
                        1,
                        0.024,
                        transform=ax.transAxes,
                        color=bg_col,
                        zorder=2,
                    )
                )
                ax.text(
                    0.04,
                    y_pos - 0.011,
                    label,
                    transform=ax.transAxes,
                    ha="left",
                    va="center",
                    fontsize=10.0,
                    color="#2c3e50",
                    fontweight="bold",
                    zorder=3,
                )
                ax.text(
                    0.96,
                    y_pos - 0.011,
                    str(value),
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=10.0,
                    color="#1a252f",
                    zorder=3,
                )
                return y_pos - 0.024

            # Main Card Header
            ax.add_patch(
                patches.Rectangle(
                    (0, y - 0.045), 1, 0.052, transform=ax.transAxes, color=c_header, zorder=2
                )
            )
            ax.text(
                0.5,
                y - 0.021,
                "FedMedAI — Federated Training Summary Card",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=13.5,
                fontweight="bold",
                color="white",
                zorder=3,
            )
            y -= 0.056

            # Section 1: Experiment & Model Setup
            y = draw_sec_header(ax, y, "1. EXPERIMENT & MODEL SETUP", c_sec_a)
            r_idx = 0

            def next_bg():
                nonlocal r_idx
                bg = c_row_light if r_idx % 2 == 0 else c_row_dark
                r_idx += 1
                return bg

            y = draw_entry(ax, y, "Strategy Algorithm", self.strategy_name.upper(), next_bg())
            if self.proximal_mu is not None:
                y = draw_entry(ax, y, "FedProx Proximal Mu (μ)", str(self.proximal_mu), next_bg())

            # Model Architecture info
            m_name_disp = (self.model_name or "CNN").upper()
            y = draw_entry(ax, y, "Model Architecture", m_name_disp, next_bg())

            # Total Parameters
            if self.total_parameters is not None:
                tr_info = f" (Trainable: {self.trainable_parameters:,})" if self.trainable_parameters is not None else ""
                if self.total_parameters >= 1_000_000:
                    param_str = f"{self.total_parameters:,} (~{self.total_parameters/1e6:.2f}M){tr_info}"
                elif self.total_parameters >= 1_000:
                    param_str = f"{self.total_parameters:,} (~{self.total_parameters/1e3:.1f}K){tr_info}"
                else:
                    param_str = f"{self.total_parameters:,}{tr_info}"
            else:
                param_str = "N/A"
            y = draw_entry(ax, y, "Total Model Parameters", param_str, next_bg())

            # Model Weight Payload & Checkpoint File Size
            if self.model_weight_size_kb is not None:
                weight_str = f"{self.model_weight_size_kb:.2f} KB ({self.model_weight_size_kb/1024.0:.3f} MB)"
            else:
                weight_str = "N/A"
            y = draw_entry(ax, y, "Model Weights (In-Memory Payload)", weight_str, next_bg())

            if self.model_file_size_kb is not None:
                file_str = f"{self.model_file_size_kb:.2f} KB ({self.model_file_size_kb/1024.0:.3f} MB)"
            else:
                file_str = weight_str
            y = draw_entry(ax, y, "Model Checkpoint File (.pth)", file_str, next_bg())

            y = draw_entry(ax, y, "Total Rounds Completed", str(len(self.round_records)), next_bg())
            y = draw_entry(ax, y, "Active Clients Reporting", str(len(client_ids)), next_bg())
            device_types = sorted(list(set(c["device_type"] for c in self.client_records)))
            y = draw_entry(ax, y, "Device Classes Present", ", ".join(device_types) if device_types else "N/A", next_bg())
            y -= 0.010

            # Section 2: Model Performance & Benchmark Metrics
            y = draw_sec_header(ax, y, "2. MODEL PERFORMANCE & ACCURACY", c_sec_b)
            r_idx = 0

            # BMCTA and BMCTA Loss (FedSLD Paper & Client Local Test Benchmark)
            if self.local_test_enabled or self.bmcta > 0:
                bmcta_disp = (
                    f"{self.bmcta*100:.2f}% (Round {self.bmcta_round})"
                    if self.bmcta > 0
                    else "N/A"
                )
                y = draw_entry(ax, y, "BMCTA (Best Mean Client Test Acc)", bmcta_disp, next_bg())

                bmcta_loss_disp = (
                    f"{self.bmcta_loss:.4f} (Round {self.bmcta_round})"
                    if self.bmcta_loss is not None
                    else "N/A"
                )
                y = draw_entry(ax, y, "BMCTA Loss (Best Client Eval Loss)", bmcta_loss_disp, next_bg())

                final_c_test_acc = (
                    f"{self.round_records[-1]['mcta']*100:.2f}%"
                    if self.round_records and self.round_records[-1].get("mcta") is not None
                    else "N/A"
                )
                final_c_test_loss = (
                    f"{self.round_records[-1]['mctl']:.4f}"
                    if self.round_records and self.round_records[-1].get("mctl") is not None
                    else "N/A"
                )
                y = draw_entry(ax, y, "Final Client Test Accuracy (Avg)", final_c_test_acc, next_bg())
                y = draw_entry(ax, y, "Final Client Test Loss (Eval Loss Avg)", final_c_test_loss, next_bg())

                bta_disp = (
                    f"{self.bta*100:.2f}% (Round {self.bta_round})"
                    if self.bta > 0
                    else "N/A"
                )
                y = draw_entry(ax, y, "BTA (Best Test Accuracy - Peak Client)", bta_disp, next_bg())

            # Global Server Evaluation
            fedbn_strats = ("fedbn", "proposed", "fedbns", "fedbn_scaffold", "bn_scaffold", "scaffold_fedbn")
            is_bn_strat = self.strategy_name.lower() in fedbn_strats

            if is_bn_strat and self.best_eval_accuracy == 0:
                best_eval_acc = "Bypassed (Local BN retained on clients)"
                final_eval_acc = "Bypassed (Client local test evaluated)"
                best_eval_loss = "Bypassed"
                final_eval_loss = "Bypassed"
            else:
                best_eval_acc = (
                    f"{self.best_eval_accuracy*100:.2f}% (Round {self.best_eval_round})"
                    if self.best_eval_accuracy > 0
                    else "N/A"
                )
                best_eval_loss = (
                    f"{self.best_eval_loss:.4f} (Round {self.best_eval_round})"
                    if self.best_eval_loss != float("inf")
                    else "N/A"
                )
                final_eval_acc = (
                    f"{self.round_records[-1]['eval_accuracy']*100:.2f}%"
                    if self.round_records and self.round_records[-1].get("eval_accuracy") is not None
                    else "N/A"
                )
                final_eval_loss = (
                    f"{self.round_records[-1]['eval_loss']:.4f}"
                    if self.round_records and self.round_records[-1].get("eval_loss") is not None
                    else "N/A"
                )

            y = draw_entry(ax, y, "Best Global Test Accuracy", best_eval_acc, next_bg())
            y = draw_entry(ax, y, "Best Global Test Loss (Eval Loss)", best_eval_loss, next_bg())
            y = draw_entry(ax, y, "Final Global Test Accuracy", final_eval_acc, next_bg())
            y = draw_entry(ax, y, "Final Global Test Loss (Eval Loss)", final_eval_loss, next_bg())

            final_train_acc = (
                f"{self.round_records[-1]['train_accuracy_avg']*100:.2f}%"
                if self.round_records
                else "N/A"
            )
            final_train_loss = (
                f"{self.round_records[-1]['train_loss_avg']:.4f}"
                if self.round_records
                else "N/A"
            )
            y = draw_entry(ax, y, "Final Client Train Accuracy (Avg)", final_train_acc, next_bg())
            y = draw_entry(ax, y, "Final Client Train Loss (Avg)", final_train_loss, next_bg())
            y -= 0.010

            # Section 3: Time & System Heterogeneity
            y = draw_sec_header(ax, y, "3. SYSTEM TIMING & HETEROGENEITY", "#2d7f5e")
            r_idx = 0
            total_wall_time = (
                self.round_records[-1]["total_elapsed_seconds"]
                if self.round_records
                else (time.time() - self.fl_start_time)
            )
            round_times = [r["round_time_seconds"] for r in self.round_records]
            avg_r_time = np.mean(round_times) if round_times else 0.0
            straggler_delays = [r["straggler_time_seconds"] for r in self.round_records]
            total_straggler = sum(straggler_delays)
            straggler_pct = (
                (total_straggler / total_wall_time * 100) if total_wall_time > 0 else 0.0
            )

            y = draw_entry(
                ax,
                y,
                "Total Wall-Clock Time",
                f"{total_wall_time:.1f}s ({total_wall_time/60:.2f} min)",
                next_bg(),
            )
            y = draw_entry(
                ax,
                y,
                "Average Round Duration",
                f"{avg_r_time:.2f}s (Min: {min(round_times):.1f}s, Max: {max(round_times):.1f}s)" if round_times else "0.0s",
                next_bg(),
            )
            y = draw_entry(
                ax,
                y,
                "Total Straggler Latency Penalty",
                f"{total_straggler:.1f}s ({straggler_pct:.1f}% of wall time)",
                next_bg(),
            )

            # Per client compute speeds
            c_avg_times = [
                (
                    cid,
                    np.mean(
                        [
                            c["training_time_seconds"]
                            for c in self.client_records
                            if c["client_id"] == cid
                        ]
                    ),
                )
                for cid in client_ids
            ]
            if c_avg_times:
                fastest_c = min(c_avg_times, key=lambda x: x[1])
                slowest_c = max(c_avg_times, key=lambda x: x[1])
                y = draw_entry(
                    ax,
                    y,
                    "Fastest Client Compute (Avg)",
                    f"Client {fastest_c[0]}: {fastest_c[1]:.2f}s / round",
                    next_bg(),
                )
                y = draw_entry(
                    ax,
                    y,
                    "Slowest Client Compute (Avg)",
                    f"Client {slowest_c[0]}: {slowest_c[1]:.2f}s / round",
                    next_bg(),
                )
                speedup = slowest_c[1] / max(0.001, fastest_c[1])
                y = draw_entry(
                    ax, y, "Compute Disparity Ratio", f"{speedup:.2f}x speed gap", next_bg()
                )
            y -= 0.010

            # Section 4: Communication & Artifacts
            y = draw_sec_header(ax, y, "4. COMMUNICATION & ARTIFACTS", "#8e44ad")
            r_idx = 0
            total_comm_kb = sum(r["total_weight_size_kb"] for r in self.round_records)
            per_client_up = (
                f"{self.model_weight_size_kb:.2f} KB / round"
                if self.model_weight_size_kb is not None
                else "N/A"
            )
            y = draw_entry(ax, y, "Model Weight Upload per Client", per_client_up, next_bg())
            y = draw_entry(
                ax,
                y,
                "Total Client Upload Data",
                f"{total_comm_kb/1024:.2f} MB ({total_comm_kb:.0f} KB)",
                next_bg(),
            )

            all_pings = [c["ping_ms"] for c in self.client_records if c.get("ping_ms") is not None]
            if all_pings:
                avg_p = float(np.mean(all_pings))
                min_p = float(np.min(all_pings))
                max_p = float(np.max(all_pings))
                y = draw_entry(
                    ax,
                    y,
                    "Client-to-Server Ping (Avg)",
                    f"{avg_p:.2f} ms (Min: {min_p:.2f} ms, Max: {max_p:.2f} ms)",
                    next_bg(),
                )

            y = draw_entry(
                ax,
                y,
                "Output Data Formats",
                "3 CSV tables (Client, Epoch, Round) + fl_results.json",
                next_bg(),
            )
            y = draw_entry(
                ax,
                y,
                "Comparison Visualizations",
                "5 publication-quality PNG charts + Summary Card",
                next_bg(),
            )

            # Footer
            ax.text(
                0.5,
                0.015,
                f"Generated by FedMedAI Monitoring • {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8.5,
                color="#7f8c8d",
                fontstyle="italic",
            )

            out_p = os.path.join(save_dir, "fl_summary_card.png")
            plt.savefig(out_p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            generated_plots.append(out_p)
        except Exception as e:
            print(f"[MetricsRecorder] Error plotting summary card: {e}")

        # Parameter Dynamics and Weight Changes Diagnostic Plot
        if self.parameter_round_records:
            try:
                param_plot_p = self.plot_parameter_changes(save_dir)
                if param_plot_p:
                    generated_plots.append(param_plot_p)
            except Exception as e:
                print(f"[MetricsRecorder] Error plotting parameter dynamics: {e}")

        return generated_plots

    def plot_parameter_changes(self, save_dir: str) -> Optional[str]:
        """Generate a 4-panel publication-grade plot of parameter updates and client weight drift."""
        if not self.parameter_round_records:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        rounds = [r["round"] for r in self.parameter_round_records]

        # 1. Global Model Update Step Size (L2 Delta)
        ax = axes[0, 0]
        deltas = [r["global_delta_l2"] for r in self.parameter_round_records]
        ax.plot(rounds, deltas, marker="o", color="#2563eb", linewidth=2.0, label=r"$\|\Delta W_{global}\|_2$")
        ax.set_title("Global Model Update Step Magnitude per Round", fontsize=11, fontweight="bold")
        ax.set_xlabel("Federated Round", fontsize=10)
        ax.set_ylabel(r"L2 Norm of Update $\|\Delta W\|_2$", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=9)

        # 2. Client Parameter Drift from Global Model
        ax = axes[0, 1]
        avg_drift = [r["avg_client_drift_l2"] for r in self.parameter_round_records]
        max_drift = [r["max_client_drift_l2"] for r in self.parameter_round_records]
        min_drift = [r["min_client_drift_l2"] for r in self.parameter_round_records]
        ax.plot(rounds, avg_drift, marker="s", color="#16a34a", linewidth=2.0, label="Mean Client Drift")
        ax.fill_between(rounds, min_drift, max_drift, color="#16a34a", alpha=0.2, label="Min-Max Client Range")
        ax.set_title(r"Client Parameter Drift from Global ($\|W_{client} - W_{global}^{t-1}\|_2$)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Federated Round", fontsize=10)
        ax.set_ylabel("Drift Distance (L2)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=9)

        # 3. Global Model Parameter Vector L2 Norm
        ax = axes[1, 0]
        norms = [r["global_l2_norm"] for r in self.parameter_round_records]
        ax.plot(rounds, norms, marker="^", color="#9333ea", linewidth=2.0, label=r"$\|W_{global}\|_2$")
        ax.set_title("Global Model Total Weight L2 Norm", fontsize=11, fontweight="bold")
        ax.set_xlabel("Federated Round", fontsize=10)
        ax.set_ylabel(r"Parameter Vector Norm $\|W\|_2$", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=9)

        # 4. Client Divergence from Consensus Model
        ax = axes[1, 1]
        avg_div = [r["avg_client_divergence_l2"] for r in self.parameter_round_records]
        max_div = [r["max_client_divergence_l2"] for r in self.parameter_round_records]
        ax.plot(rounds, avg_div, marker="d", color="#ea580c", linewidth=2.0, label="Mean Divergence from Consensus")
        ax.plot(rounds, max_div, linestyle="--", color="#dc2626", linewidth=1.5, label="Max Client Divergence")
        ax.set_title(r"Client Divergence from Aggregated Model ($\|W_{client} - W_{agg}^{t}\|_2$)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Federated Round", fontsize=10)
        ax.set_ylabel("Divergence Distance (L2)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=9)

        fig.suptitle(f"Parameter Dynamics & Weight Changes Diagnostic ({self.strategy_name.upper()})", fontsize=13, fontweight="bold", y=0.995)
        plt.tight_layout()
        out_p = os.path.join(save_dir, "parameter_changes_plot.png")
        plt.savefig(out_p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_p

    # ───────────────────────────────────────────────────────────
    # Master Save Method
    # ───────────────────────────────────────────────────────────

    def save_all(
        self,
        save_dir: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Save all CSV files, comparison plots, and rich fl_results.json."""
        os.makedirs(save_dir, exist_ok=True)

        # 1. Resolve model metadata if passed via extra_metadata or inferrable
        if extra_metadata:
            m_name = extra_metadata.get("model") or extra_metadata.get("model_name")
            if m_name and not self.model_name:
                self.set_model_info(model_name=m_name)

        # 2. Check for checkpoint files (.pth) already saved on disk
        best_pth = os.path.join(save_dir, "best_model.pth")
        final_pth = os.path.join(save_dir, "final_model.pth")
        if os.path.exists(best_pth):
            self.model_file_size_kb = os.path.getsize(best_pth) / 1024.0
        elif os.path.exists(final_pth):
            self.model_file_size_kb = os.path.getsize(final_pth) / 1024.0

        # 3. Infer weight size from client records if not already populated
        if self.model_weight_size_kb is None and self.client_records:
            c_weights = [
                c["weight_size_kb"]
                for c in self.client_records
                if c.get("weight_size_kb") is not None and c["weight_size_kb"] > 0
            ]
            if c_weights:
                self.model_weight_size_kb = float(np.median(c_weights))

        # 4. Infer parameters from model name if not already computed
        if self.total_parameters is None and self.model_name:
            self.set_model_info(model_name=self.model_name)

        csv_paths = self.export_csvs(save_dir)
        plot_paths = self.generate_plots(save_dir)
        json_path = self.export_json(
            save_dir, extra_metadata=extra_metadata, plot_files=plot_paths
        )

        return {
            "save_dir": save_dir,
            "json_path": json_path,
            "csv_paths": csv_paths,
            "plot_paths": plot_paths,
            "summary": self.get_summary(),
        }
