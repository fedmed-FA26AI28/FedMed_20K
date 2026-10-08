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
    ):
        self.strategy_name = strategy_name
        self.proximal_mu = proximal_mu
        self.fl_start_time = time.time()

        self.round_start_times: Dict[int, float] = {}
        self.round_records: List[Dict[str, Any]] = []
        self.client_records: List[Dict[str, Any]] = []
        self.epoch_records: List[Dict[str, Any]] = []
        self.eval_records: Dict[int, Dict[str, Any]] = {}

        self.best_eval_accuracy: float = 0.0
        self.best_eval_round: int = 0
        self.best_eval_loss: float = float("inf")

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
            val_loss = (
                float(m["val_loss"]) if m.get("val_loss") is not None else None
            )
            val_accuracy = (
                float(m["val_accuracy"])
                if m.get("val_accuracy") is not None
                else None
            )
            num_val_samples = int(m.get("num_val_samples", 0))
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
                "val_loss": round(val_loss, 6) if val_loss is not None else None,
                "val_accuracy": (
                    round(val_accuracy, 6) if val_accuracy is not None else None
                ),
                "num_val_samples": num_val_samples,
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

        if acc_val is not None and acc_val > self.best_eval_accuracy:
            self.best_eval_accuracy = acc_val
            self.best_eval_round = server_round
            if loss_val is not None:
                self.best_eval_loss = loss_val

        # Update existing round record if already present
        for r in self.round_records:
            if r["round"] == server_round:
                r["eval_loss"] = round(loss_val, 6) if loss_val is not None else None
                r["eval_accuracy"] = round(acc_val, 6) if acc_val is not None else None
                break

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

        summary = {
            "strategy": self.strategy_name,
            "total_rounds": len(self.round_records),
            "total_time_seconds": round(total_time, 2),
            "avg_round_time_seconds": round(avg_round, 2),
            "best_metric": round(best_metric, 4),
            "best_round": self.best_eval_round,
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
            "val_loss",
            "val_accuracy",
            "num_val_samples",
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
                    "val_loss": c.get("val_loss"),
                    "val_accuracy": c.get("val_accuracy"),
                    "num_val_samples": c.get("num_val_samples", 0),
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
                "final_validation_accuracy": final_eval_acc,
                "best_global_accuracy": round(self.best_eval_accuracy, 4)
                if self.best_eval_accuracy > 0
                else None,
                "best_validation_accuracy": round(self.best_eval_accuracy, 4)
                if self.best_eval_accuracy > 0
                else None,
                "best_global_accuracy_round": self.best_eval_round
                if self.best_eval_accuracy > 0
                else None,
                "final_global_loss": final_eval_loss,
                "final_validation_loss": final_eval_loss,
                "best_global_loss": round(self.best_eval_loss, 4)
                if self.best_eval_loss != float("inf")
                else None,
                "final_train_loss_avg": final_train_loss,
                "final_train_accuracy_avg": final_train_acc,
                "total_communication_kb": round(total_comm_kb, 2),
                "total_communication_mb": round(total_comm_kb / 1024.0, 2),
            },
            "client_summaries": client_summaries,
            "hardware_heterogeneity": heterogeneity_stats,
            "rounds_history": rounds_history,
            "artifacts": {
                "client_round_metrics_csv": "client_round_metrics.csv",
                "client_epoch_metrics_csv": "client_epoch_metrics.csv",
                "round_metrics_csv": "round_metrics.csv",
                "plots": [os.path.basename(p) for p in (plot_files or [])],
            },
        }

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

            # (A) Loss over Rounds (Train vs Validation)
            ax = axes[0]
            train_losses = [r["train_loss_avg"] for r in self.round_records]
            ax.plot(rounds, train_losses, "b-o", linewidth=2, label="Client Train Loss (Avg)")
            eval_losses = [r["eval_loss"] for r in self.round_records]
            if any(l is not None for l in eval_losses):
                valid_r = [r for r, l in zip(rounds, eval_losses) if l is not None]
                valid_l = [l for l in eval_losses if l is not None]
                ax.plot(valid_r, valid_l, "r-s", linewidth=2, label="Validation Loss")
            ax.set_title("Loss over FL Rounds", fontsize=13, fontweight="bold")
            ax.set_xlabel("FL Round", fontsize=11)
            ax.set_ylabel("Loss", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(loc="best", fontsize=10)

            # (B) Accuracy over Rounds (Train vs Validation)
            ax = axes[1]
            train_accs = [r["train_accuracy_avg"] for r in self.round_records]
            ax.plot(rounds, train_accs, "b-o", linewidth=2, label="Client Train Acc (Avg)")
            eval_accs = [r["eval_accuracy"] for r in self.round_records]
            if any(a is not None for a in eval_accs):
                valid_r = [r for r, a in zip(rounds, eval_accs) if a is not None]
                valid_a = [a for a in eval_accs if a is not None]
                ax.plot(valid_r, valid_a, "g-^", linewidth=2, label="Validation Acc")
                if self.best_eval_accuracy > 0:
                    ax.axhline(
                        y=self.best_eval_accuracy,
                        color="darkgreen",
                        linestyle=":",
                        label=f"Best Val: {self.best_eval_accuracy:.4f}",
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
            fig, ax = plt.subplots(figsize=(10, 12))
            ax.axis("off")

            # Colors
            c_header = "#1b2838"
            c_sec_a = "#1f4e78"
            c_sec_b = "#2e75b6"
            c_row_light = "#f2f7fb"
            c_row_dark = "#ffffff"

            y = 0.98

            def draw_sec_header(ax, y_pos, title, col):
                ax.add_patch(
                    patches.Rectangle(
                        (0, y_pos - 0.03),
                        1,
                        0.035,
                        transform=ax.transAxes,
                        color=col,
                        zorder=2,
                    )
                )
                ax.text(
                    0.5,
                    y_pos - 0.012,
                    title,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color="white",
                    zorder=3,
                )
                return y_pos - 0.035

            def draw_entry(ax, y_pos, label, value, bg_col):
                ax.add_patch(
                    patches.Rectangle(
                        (0, y_pos - 0.026),
                        1,
                        0.028,
                        transform=ax.transAxes,
                        color=bg_col,
                        zorder=2,
                    )
                )
                ax.text(
                    0.05,
                    y_pos - 0.013,
                    label,
                    transform=ax.transAxes,
                    ha="left",
                    va="center",
                    fontsize=10.5,
                    color="#2c3e50",
                    fontweight="bold",
                    zorder=3,
                )
                ax.text(
                    0.95,
                    y_pos - 0.013,
                    str(value),
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=10.5,
                    color="#1a252f",
                    zorder=3,
                )
                return y_pos - 0.028

            # Main Card Header
            ax.add_patch(
                patches.Rectangle(
                    (0, y - 0.05), 1, 0.06, transform=ax.transAxes, color=c_header, zorder=2
                )
            )
            ax.text(
                0.5,
                y - 0.025,
                f"FedMedAI — Federated Training Summary Card",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="white",
                zorder=3,
            )
            y -= 0.065

            # Section 1: Experiment Setup
            y = draw_sec_header(ax, y, "1. EXPERIMENT SETUP", c_sec_a)
            y = draw_entry(ax, y, "Strategy Algorithm", self.strategy_name.upper(), c_row_light)
            if self.proximal_mu is not None:
                y = draw_entry(ax, y, "FedProx Proximal Mu (μ)", str(self.proximal_mu), c_row_dark)
            y = draw_entry(ax, y, "Total Rounds Completed", str(len(self.round_records)), c_row_light)
            y = draw_entry(ax, y, "Active Clients Reporting", str(len(client_ids)), c_row_dark)
            device_types = sorted(list(set(c["device_type"] for c in self.client_records)))
            y = draw_entry(ax, y, "Device Classes Present", ", ".join(device_types), c_row_light)
            y -= 0.015

            # Section 2: Model Performance
            y = draw_sec_header(ax, y, "2. MODEL PERFORMANCE", c_sec_b)
            final_eval_acc = (
                f"{self.round_records[-1]['eval_accuracy']*100:.2f}%"
                if self.round_records and self.round_records[-1]["eval_accuracy"] is not None
                else "N/A"
            )
            best_eval_acc = (
                f"{self.best_eval_accuracy*100:.2f}% (Round {self.best_eval_round})"
                if self.best_eval_accuracy > 0
                else "N/A"
            )
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

            y = draw_entry(ax, y, "Best Global Validation Accuracy", best_eval_acc, c_row_light)
            y = draw_entry(ax, y, "Final Global Validation Accuracy", final_eval_acc, c_row_dark)
            y = draw_entry(ax, y, "Final Client Train Accuracy (Avg)", final_train_acc, c_row_light)
            y = draw_entry(ax, y, "Final Client Train Loss (Avg)", final_train_loss, c_row_dark)
            y -= 0.015

            # Section 3: Time & System Heterogeneity
            y = draw_sec_header(ax, y, "3. SYSTEM TIMING & HETEROGENEITY", "#2d7f5e")
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
                c_row_light,
            )
            y = draw_entry(
                ax,
                y,
                "Average Round Duration",
                f"{avg_r_time:.2f}s (Min: {min(round_times):.1f}s, Max: {max(round_times):.1f}s)",
                c_row_dark,
            )
            y = draw_entry(
                ax,
                y,
                "Total Straggler Latency Penalty",
                f"{total_straggler:.1f}s ({straggler_pct:.1f}% of wall time)",
                c_row_light,
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
                    c_row_dark,
                )
                y = draw_entry(
                    ax,
                    y,
                    "Slowest Client Compute (Avg)",
                    f"Client {slowest_c[0]}: {slowest_c[1]:.2f}s / round",
                    c_row_light,
                )
                speedup = slowest_c[1] / max(0.001, fastest_c[1])
                y = draw_entry(
                    ax, y, "Compute Disparity Ratio", f"{speedup:.2f}x speed gap", c_row_dark
                )
            y -= 0.015

            # Section 4: Communication & Artifacts
            y = draw_sec_header(ax, y, "4. COMMUNICATION & OUTPUT ARTIFACTS", "#8e44ad")
            total_comm_kb = sum(r["total_weight_size_kb"] for r in self.round_records)
            y = draw_entry(
                ax,
                y,
                "Total Client Upload Data",
                f"{total_comm_kb/1024:.2f} MB ({total_comm_kb:.0f} KB)",
                c_row_light,
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
                    c_row_dark,
                )

            y = draw_entry(
                ax,
                y,
                "Output Data Formats",
                "3 CSV tables (Client, Epoch, Round) + fl_results.json",
                c_row_light if all_pings else c_row_dark,
            )
            y = draw_entry(
                ax,
                y,
                "Comparison Visualizations",
                "5 publication-quality PNG charts",
                c_row_dark if all_pings else c_row_light,
            )

            # Footer
            ax.text(
                0.5,
                0.02,
                f"Generated by FedMedAI Monitoring • {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="#7f8c8d",
                fontstyle="italic",
            )

            out_p = os.path.join(save_dir, "fl_summary_card.png")
            plt.savefig(out_p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            generated_plots.append(out_p)
        except Exception as e:
            print(f"[MetricsRecorder] Error plotting summary card: {e}")

        return generated_plots

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
