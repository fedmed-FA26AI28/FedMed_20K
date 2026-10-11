"""Multi-Strategy & Multi-Partition Simulation Runner for FedMedAI.

Executes a grid of federated learning experiments across multiple aggregation strategies,
data partition modes, and datasets.
"""

import argparse
import datetime
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.medmnist_code import get_medmnist_datasets
from datasets.partition import dirichlet_partition, pathological_partition, practical_partition, save_partition, visualize_partition


# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions: Network, Sockets & Environment
# ──────────────────────────────────────────────────────────────────────────────

def get_free_port() -> int:
    """Find and return an available ephemeral TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_server(host: str, port: int, timeout: float = 20.0) -> bool:
    """Poll server socket until it starts accepting TCP connections."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.3)
    return False


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration dictionary from YAML file."""
    if not os.path.exists(config_path):
        print(f"[Warning] Config file not found at '{config_path}'. Using empty defaults.")
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ──────────────────────────────────────────────────────────────────────────────
# Data Partition Validation & Auto-Generation
# ──────────────────────────────────────────────────────────────────────────────

def ensure_partition_exists(
    dataset_name: str,
    partition_mode: str,
    num_clients: int,
    seed: int = 42,
    partition_dir: str = "data/partitions",
) -> Path:
    """Ensure partition JSON exists for (dataset_name, partition_mode, num_clients).

    If missing, automatically generates the partition.
    """
    part_dir = Path(partition_dir)
    part_dir.mkdir(parents=True, exist_ok=True)
    part_file = part_dir / f"partition_{dataset_name}_{partition_mode}_seed{seed}_clients{num_clients}.json"

    if part_file.exists():
        return part_file

    print(f"\n[Partition] Missing partition: {part_file.name}. Generating automatically...")
    train_dataset, _, _, _ = get_medmnist_datasets(dataset_name=dataset_name, download=True)
    
    if partition_mode == "pathological":
        partition = pathological_partition(train_dataset, num_clients, seed)
    elif partition_mode == "practical":
        partition = practical_partition(train_dataset, num_clients, seed)
    elif partition_mode.startswith("dirichlet_"):
        alpha = float(partition_mode.split("_")[1])
        partition = dirichlet_partition(train_dataset, num_clients, alpha, seed)
    else:
        raise ValueError(f"Unknown partition mode: {partition_mode}")

    # Validate integrity
    total_assigned = sum(len(indices) for indices in partition)
    assert total_assigned == len(train_dataset), "Partition sample count mismatch!"
    all_indices = [idx for sublist in partition for idx in sublist]
    assert len(all_indices) == len(set(all_indices)), "Found duplicate sample indices across clients!"

    # Save mapping and visualization
    save_partition(partition, train_dataset, dataset_name, partition_mode, seed, num_clients, save_dir=str(part_dir))
    plot_path = part_dir / "plots" / f"dist_{dataset_name}_{partition_mode}_clients{num_clients}.png"
    visualize_partition(partition, train_dataset, dataset_name, partition_mode, num_clients, save_path=str(plot_path))
    print(f"[Partition] Generated and verified: {part_file} (plot: {plot_path})")

    return part_file


# ──────────────────────────────────────────────────────────────────────────────
# Single FL Experiment Runner (Subprocess Architecture)
# ──────────────────────────────────────────────────────────────────────────────

def run_single_experiment(
    strategy: str,
    partition_mode: str,
    dataset_name: str,
    partition_path: Path,
    num_clients: int,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    lr_mode: str,
    client_lr_patience: int,
    client_lr_factor: float,
    client_lr_min: float,
    lr_decay_steps: int,
    lr_decay_gamma: float,
    proximal_mu: float,
    early_stop_patience: int,
    early_stop_metric: str,
    server_eval: bool,
    device_type: str,
    save_client_metrics: bool,
    timeout_per_run: int,
    run_dir: Path,
    config_path: str,
    num_workers: int = 0,
    client_cpu_threads: int = 0,
    python_exe: str = sys.executable,
) -> Dict[str, Any]:
    """Execute a single FL training run (1 Server + N Clients) as isolated subprocesses."""
    run_dir.mkdir(parents=True, exist_ok=True)
    server_log_path = run_dir / "server.log"
    clients_dir = run_dir / "clients"
    clients_dir.mkdir(parents=True, exist_ok=True)

    port = get_free_port()
    host = "127.0.0.1"

    print(f"\n{'-'*65}")
    print(f"  Launching FL Server: {strategy.upper()} | Dataset: {dataset_name} | Partition: {partition_mode} | Port: {port}")
    print(f"  Artifact Directory : {run_dir}")
    print(f"{'-'*65}")

    # Build server command
    server_cmd = [
        python_exe, "-m", "server.server",
        "--config", str(config_path),
        "--strategy", strategy,
        "--dataset", dataset_name,
        "--rounds", str(rounds),
        "--min_clients", str(num_clients),
        "--local_epochs", str(local_epochs),
        "--learning_rate", str(learning_rate),
        "--lr_mode", lr_mode,
        "--client_lr_patience", str(client_lr_patience),
        "--client_lr_factor", str(client_lr_factor),
        "--client_lr_min", str(client_lr_min),
        "--lr_decay_steps", str(lr_decay_steps),
        "--lr_decay_gamma", str(lr_decay_gamma),
        "--proximal_mu", str(proximal_mu),
        "--early_stop_patience", str(early_stop_patience),
        "--early_stop_metric", early_stop_metric,
        "--host", host,
        "--port", str(port),
        "--save_dir", str(run_dir),
    ]
    if server_eval:
        server_cmd.append("--server_eval")
    else:
        server_cmd.append("--no-server_eval")

    server_log_file = open(server_log_path, "w", encoding="utf-8")
    
    env = os.environ.copy()
    if client_cpu_threads > 0:
        env["OMP_NUM_THREADS"] = str(client_cpu_threads)
        env["MKL_NUM_THREADS"] = str(client_cpu_threads)
        env["OPENBLAS_NUM_THREADS"] = str(client_cpu_threads)

    server_proc = subprocess.Popen(
        server_cmd,
        cwd=str(PROJECT_ROOT),
        stdout=server_log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )

    client_procs = []
    client_log_files = []
    run_start_time = time.time()
    status = "failed"
    run_metrics: Dict[str, Any] = {}

    try:
        if not wait_for_server(host, port, timeout=25.0):
            raise TimeoutError(f"Server on {host}:{port} did not start listening within timeout.")

        print(f"  [Server] Listening. Launching {num_clients} client workers...")

        for cid in range(num_clients):
            c_dir = clients_dir / f"client_{cid}"
            c_dir.mkdir(parents=True, exist_ok=True)
            c_log_path = c_dir / "client.log"
            c_log_file = open(c_log_path, "w", encoding="utf-8")
            client_log_files.append(c_log_file)

            client_cmd = [
                python_exe, "-m", "client.client",
                "--config", str(config_path),
                "--dataset", dataset_name,
                "--client_id", str(cid),
                "--server_address", f"{host}:{port}",
                "--partition_path", str(partition_path),
                "--partition_mode", partition_mode,
                "--num_clients", str(num_clients),
                "--local_epochs", str(local_epochs),
                "--learning_rate", str(learning_rate),
                "--lr_mode", lr_mode,
                "--lr_patience", str(client_lr_patience),
                "--lr_factor", str(client_lr_factor),
                "--lr_min", str(client_lr_min),
                "--batch_size", str(batch_size),
                "--num_workers", str(num_workers),
                "--device_type", device_type,
                "--strategy", strategy,
                "--client_dir", str(c_dir),
            ]
            if not save_client_metrics:
                client_cmd.append("--no_save_metrics")

            c_proc = subprocess.Popen(
                client_cmd,
                cwd=str(PROJECT_ROOT),
                stdout=c_log_file,
                stderr=subprocess.STDOUT,
                env=env,
            )
            client_procs.append(c_proc)

        loop_start = time.time()
        while True:
            server_exit = server_proc.poll()
            if server_exit is not None:
                if server_exit == 0:
                    status = "completed"
                else:
                    status = f"server_error({server_exit})"
                break

            failed_clients = []
            for c_idx, cp in enumerate(client_procs):
                c_exit = cp.poll()
                if c_exit is not None and c_exit != 0:
                    failed_clients.append((c_idx, c_exit))

            if failed_clients:
                cid, err_code = failed_clients[0]
                status = f"client_error({err_code})"
                print(f"  [Error] Client {cid} exited with error code {err_code}!")
                server_proc.kill()
                for cp in client_procs:
                    cp.kill()
                break

            if timeout_per_run and timeout_per_run > 0 and (time.time() - loop_start > timeout_per_run):
                print(f"  [Timeout] Run exceeded timeout limit of {timeout_per_run}s.")
                server_proc.kill()
                for cp in client_procs:
                    cp.kill()
                status = "timeout"
                break

            time.sleep(0.5)

        for c_proc in client_procs:
            try:
                c_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                c_proc.kill()

    except subprocess.TimeoutExpired:
        status = "timeout"
        server_proc.kill()
        for c_proc in client_procs:
            c_proc.kill()
    except KeyboardInterrupt:
        server_proc.kill()
        for c_proc in client_procs:
            c_proc.kill()
        raise
    except Exception as e:
        status = f"error: {type(e).__name__}"
        server_proc.kill()
        for c_proc in client_procs:
            c_proc.kill()
    finally:
        server_log_file.close()
        for f in client_log_files:
            f.close()

    total_wall_time = time.time() - run_start_time

    results_json_path = run_dir / "fl_results.json"
    if results_json_path.exists():
        try:
            with open(results_json_path, "r", encoding="utf-8") as f:
                fl_data = json.load(f)
            summary = fl_data.get("summary", {})
            metadata = fl_data.get("metadata", {})
            rounds_history = fl_data.get("rounds_history", [])

            run_metrics = {
                "strategy": strategy,
                "partition_mode": partition_mode,
                "dataset": dataset_name,
                "num_clients": num_clients,
                "rounds_requested": rounds,
                "rounds_completed": metadata.get("num_rounds_completed", len(rounds_history)),
                "local_epochs": local_epochs,
                "batch_size": batch_size,
                "lr_mode": lr_mode,
                "learning_rate": learning_rate,
                "final_test_accuracy": float(summary.get("final_global_accuracy", 0.0)),
                "best_test_accuracy": float(summary.get("best_global_accuracy", 0.0)),
                "best_round": int(summary.get("best_global_accuracy_round", 0)),
                "final_test_loss": float(summary.get("final_global_loss", 0.0)),
                "best_test_loss": float(summary.get("best_global_loss", 0.0)),
                "total_time_seconds": float(summary.get("total_wall_clock_time_seconds", total_wall_time)),
                "avg_round_time_seconds": float(summary.get("avg_round_time_seconds", 0.0)),
                "early_stopped": bool(summary.get("early_stopped", False) or metadata.get("early_stopped", False)),
                "status": status,
                "run_dir": str(run_dir),
                "rounds_history": rounds_history,
            }
            print(f"  [Success] Finished {strategy.upper()} ({partition_mode}): "
                  f"Best Acc: {run_metrics['best_test_accuracy']*100:.2f}% (R{run_metrics['best_round']}) | "
                  f"Time: {run_metrics['total_time_seconds']:.1f}s")
            return run_metrics
        except Exception as e:
            pass

    # Fallback: Parse server.log if fl_results.json was not generated (e.g., timeout or interrupt)
    fallback_history = []
    server_log_path = run_dir / "server.log"
    if server_log_path.exists():
        try:
            with open(server_log_path, "r", encoding="utf-8", errors="ignore") as f:
                log_text = f.read()
            pattern = re.compile(r"fit progress:\s*\(\s*(\d+),\s*([0-9.]+),\s*\{'accuracy':\s*([0-9.]+)\},\s*([0-9.]+)\)")
            for match in pattern.finditer(log_text):
                r_num = int(match.group(1))
                r_loss = float(match.group(2))
                r_acc = float(match.group(3))
                r_time = float(match.group(4))
                fallback_history.append({
                    "round": r_num,
                    "eval_accuracy": r_acc,
                    "eval_loss": r_loss,
                    "train_accuracy_avg": 0.0,
                    "train_loss_avg": 0.0,
                    "round_time_seconds": 0.0,
                    "total_elapsed_seconds": r_time,
                })
        except Exception:
            pass

    best_acc = max((h["eval_accuracy"] for h in fallback_history), default=0.0)
    best_r = next((h["round"] for h in fallback_history if h["eval_accuracy"] == best_acc), 0) if fallback_history else 0
    final_acc = fallback_history[-1]["eval_accuracy"] if fallback_history else 0.0
    final_loss = fallback_history[-1]["eval_loss"] if fallback_history else 0.0

    run_metrics = {
        "strategy": strategy,
        "partition_mode": partition_mode,
        "dataset": dataset_name,
        "num_clients": num_clients,
        "rounds_requested": rounds,
        "rounds_completed": len(fallback_history),
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "lr_mode": lr_mode,
        "learning_rate": learning_rate,
        "final_test_accuracy": final_acc,
        "best_test_accuracy": best_acc,
        "best_round": best_r,
        "final_test_loss": final_loss,
        "best_test_loss": min((h["eval_loss"] for h in fallback_history), default=0.0),
        "total_time_seconds": total_wall_time,
        "avg_round_time_seconds": (total_wall_time / max(len(fallback_history), 1)) if fallback_history else 0.0,
        "early_stopped": False,
        "status": status,
        "run_dir": str(run_dir),
        "rounds_history": fallback_history,
    }
    if fallback_history:
        print(f"  [Partial Recovery] Recovered {len(fallback_history)} rounds from server.log: "
              f"Best Acc: {best_acc*100:.2f}% (R{best_r}) | Time: {total_wall_time:.1f}s")
    return run_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Visualization & Comparison Artifact Generators
# ──────────────────────────────────────────────────────────────────────────────

STRATEGY_COLORS = {
    "fedavg": "#2563eb",
    "fedprox": "#16a34a",
    "fednova": "#ea580c",
    "fedbn": "#9333ea",
    "scaffold": "#dc2626",
    "proposed": "#0891b2",
}

STRATEGY_MARKERS = {
    "fedavg": "o",
    "fedprox": "s",
    "fednova": "^",
    "fedbn": "D",
    "scaffold": "v",
    "proposed": "P",
}


def plot_accuracy_curves(df_history: pd.DataFrame, dataset_name: str, partitions: List[str], output_path: Path):
    if df_history.empty:
        return

    num_parts = len(partitions)
    fig, axes = plt.subplots(1, num_parts, figsize=(5.5 * num_parts, 4.5), sharey=True, squeeze=False)
    axes = axes[0]

    for idx, part in enumerate(partitions):
        ax = axes[idx]
        sub_df = df_history[df_history["partition_mode"] == part]

        for strat, strat_group in sub_df.groupby("strategy"):
            strat_group = strat_group.sort_values("round")
            color = STRATEGY_COLORS.get(strat.lower(), "#64748b")
            marker = STRATEGY_MARKERS.get(strat.lower(), "o")
            acc_pct = strat_group["eval_accuracy"] * 100
            ax.plot(
                strat_group["round"], acc_pct, label=strat.upper(),
                color=color, marker=marker, markersize=4.5, linewidth=1.8, alpha=0.9,
            )

        ax.set_title(f"Partition = {part}", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Federated Round", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Global Test Accuracy (%)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=8.5, loc="lower right")

    fig.suptitle(f"Accuracy Convergence Across Strategies ({dataset_name.upper()})", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(df_history: pd.DataFrame, dataset_name: str, partitions: List[str], output_path: Path):
    if df_history.empty:
        return

    num_parts = len(partitions)
    fig, axes = plt.subplots(1, num_parts, figsize=(5.5 * num_parts, 4.5), sharey=True, squeeze=False)
    axes = axes[0]

    for idx, part in enumerate(partitions):
        ax = axes[idx]
        sub_df = df_history[df_history["partition_mode"] == part]

        for strat, strat_group in sub_df.groupby("strategy"):
            strat_group = strat_group.sort_values("round")
            color = STRATEGY_COLORS.get(strat.lower(), "#64748b")
            marker = STRATEGY_MARKERS.get(strat.lower(), "o")
            ax.plot(
                strat_group["round"], strat_group["eval_loss"], label=strat.upper(),
                color=color, marker=marker, markersize=4.5, linewidth=1.8, alpha=0.9,
            )

        ax.set_title(f"Partition = {part}", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Federated Round", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Global Test Loss", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=8.5, loc="upper right")

    fig.suptitle(f"Loss Convergence Across Strategies ({dataset_name.upper()})", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_barchart(df_summary: pd.DataFrame, dataset_name: str, strategies: List[str], partitions: List[str], output_path: Path):
    if df_summary.empty:
        return

    pivot_acc = df_summary.pivot(index="partition_mode", columns="strategy", values="best_test_accuracy") * 100
    existing_strats = [s for s in strategies if s in pivot_acc.columns]
    existing_parts = [p for p in partitions if p in pivot_acc.index]
    pivot_acc = pivot_acc.reindex(index=existing_parts, columns=existing_strats)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(existing_parts))
    width = 0.8 / max(len(existing_strats), 1)

    for i, strat in enumerate(existing_strats):
        vals = pivot_acc[strat].values
        color = STRATEGY_COLORS.get(strat.lower(), "#64748b")
        rects = ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=strat.upper(), color=color, alpha=0.9, edgecolor="black", linewidth=0.5)
        for rect in rects:
            h = rect.get_height()
            if not np.isnan(h) and h > 0:
                ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_ylabel("Best Test Accuracy (%)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Partition Mode", fontsize=10, fontweight="bold")
    ax.set_title(f"Best Accuracy Comparison ({dataset_name.upper()})", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}" for p in existing_parts], fontsize=9)
    ax.set_ylim(0, min(100, max(pivot_acc.max().max() * 1.15, 50) if not pivot_acc.empty else 100))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_runtime_barchart(df_summary: pd.DataFrame, dataset_name: str, strategies: List[str], partitions: List[str], output_path: Path):
    if df_summary.empty:
        return

    pivot_time = df_summary.pivot(index="partition_mode", columns="strategy", values="total_time_seconds")
    existing_strats = [s for s in strategies if s in pivot_time.columns]
    existing_parts = [p for p in partitions if p in pivot_time.index]
    pivot_time = pivot_time.reindex(index=existing_parts, columns=existing_strats)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(existing_parts))
    width = 0.8 / max(len(existing_strats), 1)

    for i, strat in enumerate(existing_strats):
        vals = pivot_time[strat].values
        color = STRATEGY_COLORS.get(strat.lower(), "#64748b")
        rects = ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=strat.upper(), color=color, alpha=0.9, edgecolor="black", linewidth=0.5)
        for rect in rects:
            h = rect.get_height()
            if not np.isnan(h) and h > 0:
                ax.annotate(f"{h:.0f}s", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

    ax.set_ylabel("Total Wall-Clock Time (seconds)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Partition Mode", fontsize=10, fontweight="bold")
    ax.set_title(f"Runtime Comparison ({dataset_name.upper()})", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}" for p in existing_parts], fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_heatmap(df_summary: pd.DataFrame, dataset_name: str, strategies: List[str], partitions: List[str], output_path: Path):
    if df_summary.empty:
        return

    pivot_acc = df_summary.pivot(index="strategy", columns="partition_mode", values="best_test_accuracy") * 100
    existing_strats = [s for s in strategies if s in pivot_acc.index]
    existing_parts = [p for p in partitions if p in pivot_acc.columns]
    pivot_acc = pivot_acc.reindex(index=existing_strats, columns=existing_parts)

    fig, ax = plt.subplots(figsize=(6.5, max(4.0, len(existing_strats) * 0.8)))
    data = pivot_acc.values

    im = ax.imshow(data, cmap="YlGnBu", aspect="auto")

    for i in range(len(existing_strats)):
        for j in range(len(existing_parts)):
            val = data[i, j]
            if not np.isnan(val):
                text_color = "white" if val > (np.nanmax(data) + np.nanmin(data)) / 2 else "black"
                ax.text(j, i, f"{val:.2f}%", ha="center", va="center", color=text_color, fontweight="bold", fontsize=10)

    ax.set_xticks(np.arange(len(existing_parts)))
    ax.set_yticks(np.arange(len(existing_strats)))
    ax.set_xticklabels([f"{p}" for p in existing_parts], fontsize=10, fontweight="bold")
    ax.set_yticklabels([s.upper() for s in existing_strats], fontsize=10, fontweight="bold")
    ax.set_title(f"Best Test Accuracy Heatmap ({dataset_name.upper()})", fontsize=11, fontweight="bold", pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Accuracy (%)", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_markdown_report(
    df_summary: pd.DataFrame,
    strategies: List[str],
    datasets: List[str],
    partitions: List[str],
    sim_config: Dict[str, Any],
    output_path: Path,
):
    lines = [
        "# FedMedAI Simulation Benchmark Report",
        "",
        f"**Generated At:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Rounds:** {sim_config.get('rounds')} | **Clients:** {sim_config.get('num_clients')} | "
        f"**Local Epochs:** {sim_config.get('local_epochs')} | **LR Mode:** `{sim_config.get('lr_mode')}`  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
    ]

    if df_summary.empty or "dataset" not in df_summary.columns:
        lines.append("No completed simulation runs recorded.")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return

    for d in datasets:
        sub_d = df_summary[df_summary["dataset"] == d]
        if sub_d.empty:
            continue
        lines.append(f"### Dataset: **{d.upper()}**")
        for p in partitions:
            sub = sub_d[sub_d["partition_mode"] == p]
            if not sub.empty:
                best_row = sub.loc[sub["best_test_accuracy"].idxmax()]
                lines.append(f"- **Partition = {p}:** 🏆 **{best_row['strategy'].upper()}** achieved **{best_row['best_test_accuracy']*100:.2f}%** accuracy in Round {int(best_row['best_round'])} (runtime: {best_row['total_time_seconds']:.1f}s).")
        lines.append("")

    lines.extend([
        "---",
        "## 2. Full Simulation Results Matrix",
        "",
        "| Dataset | Strategy | Partition | Final Acc (%) | Best Acc (%) | Best Round | Final Loss | Total Time |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    for _, row in df_summary.iterrows():
        lines.append(f"| **{row['dataset']}** | **{row['strategy'].upper()}** | `{row['partition_mode']}` | {row['final_test_accuracy']*100:.2f}% | **{row['best_test_accuracy']*100:.2f}%** | R{int(row['best_round'])} | {row['final_test_loss']:.4f} | {row['total_time_seconds']:.1f}s |")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# Main Simulation Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FedMedAI Automated Multi-Strategy & Multi-Partition Simulation")
    parser.add_argument("--config", type=str, default="configs/simulation.yaml")
    parser.add_argument("--num_clients", type=int, default=None)
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    parser.add_argument("--partitions", type=str, nargs="+", default=None)
    parser.add_argument("--strategies", type=str, nargs="+", default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--local_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", "--lr", type=float, default=None)
    parser.add_argument("--lr_mode", type=str, choices=["client_loss", "server_decay", "fixed"], default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--timeout_per_run", type=int, default=None, help="Max run timeout in seconds (<=0 to disable)")
    parser.add_argument("--no_timeout", dest="no_timeout", action="store_true", help="Disable run timeout completely")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--client_cpu_threads", type=int, default=None)
    parser.add_argument("--server_eval", dest="server_eval", action=argparse.BooleanOptionalAction, default=None)

    args = parser.parse_args()
    cfg = load_yaml_config(args.config)

    num_clients = args.num_clients if args.num_clients is not None else cfg.get("num_clients", 12)
    datasets = args.datasets if args.datasets is not None else cfg.get("datasets", ["bloodmnist"])
    partitions = args.partitions if args.partitions is not None else cfg.get("partitions", ["dirichlet_0.3", "pathological", "practical"])
    strategies = args.strategies if args.strategies is not None else cfg.get("strategies", ["fedavg", "fedprox", "fednova", "fedbn", "scaffold"])
    rounds = args.rounds if args.rounds is not None else cfg.get("rounds", 10)
    local_epochs = args.local_epochs if args.local_epochs is not None else cfg.get("local_epochs", 3)
    batch_size = args.batch_size if args.batch_size is not None else cfg.get("batch_size", 32)
    num_workers = args.num_workers if args.num_workers is not None else cfg.get("num_workers", 0 if sys.platform == "win32" else 0)
    client_cpu_threads = args.client_cpu_threads if args.client_cpu_threads is not None else cfg.get("client_cpu_threads", 0)
    learning_rate = args.learning_rate if args.learning_rate is not None else cfg.get("learning_rate", 0.001)
    lr_mode = args.lr_mode if args.lr_mode is not None else cfg.get("lr_mode", "client_loss")
    client_lr_patience = cfg.get("client_lr_patience", 2)
    client_lr_factor = cfg.get("client_lr_factor", 0.5)
    client_lr_min = cfg.get("client_lr_min", 1e-6)
    lr_decay_steps = cfg.get("lr_decay_steps", 3)
    lr_decay_gamma = cfg.get("lr_decay_gamma", 0.5)
    proximal_mu = cfg.get("proximal_mu", 0.01)
    early_stop_patience = cfg.get("early_stop_patience", 0)
    early_stop_metric = cfg.get("early_stop_metric", "accuracy")
    server_eval = args.server_eval if args.server_eval is not None else cfg.get("server_eval", True)
    device_type = cfg.get("device_type", "pc")
    save_client_metrics = cfg.get("save_client_metrics", True)

    if getattr(args, "no_timeout", False):
        timeout_per_run = 0
    elif args.timeout_per_run is not None:
        timeout_per_run = args.timeout_per_run
    else:
        timeout_per_run = cfg.get("timeout_per_run", 0)
    if timeout_per_run is None or timeout_per_run <= 0:
        timeout_per_run = 0

    seed = cfg.get("seed", 42)

    datasets = [d.lower().strip() for d in datasets]
    partitions = [p.lower().strip() for p in partitions]
    strategies = [s.lower().strip() for s in strategies]

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        base_output_dir = Path(cfg.get("output_dir", "results/simulation")) / timestamp_str
    base_output_dir.mkdir(parents=True, exist_ok=True)

    timeout_desc = f"{timeout_per_run}s" if timeout_per_run > 0 else "Disabled (Infinite)"
    print(f"\n{'='*70}")
    print(f"  FedMedAI Automated Simulation Benchmark")
    print(f"{'='*70}")
    print(f"  Clients          : {num_clients}")
    print(f"  Rounds per Run   : {rounds}")
    print(f"  Local Epochs     : {local_epochs}")
    print(f"  Timeout per Run  : {timeout_desc}")
    print(f"  Datasets         : {datasets}")
    print(f"  Partitions       : {partitions}")
    print(f"  Strategies       : {strategies}")
    total_runs = len(datasets) * len(partitions) * len(strategies)
    print(f"  Total Runs Grid  : {total_runs} combinations")
    print(f"{'='*70}\n")

    print("Verifying data partition files for all dataset and partition combinations...")
    partition_map: Dict[Tuple[str, str], Path] = {}
    for d in datasets:
        for p in partitions:
            p_path = ensure_partition_exists(dataset_name=d, partition_mode=p, num_clients=num_clients, seed=seed)
            partition_map[(d, p)] = p_path

    results_list: List[Dict[str, Any]] = []
    rounds_history_list: List[Dict[str, Any]] = []
    current_run_idx = 0
    sim_start_time = time.time()

    for d in datasets:
        for p in partitions:
            partition_file = partition_map[(d, p)]
            for s in strategies:
                current_run_idx += 1
                run_output_dir = base_output_dir / d / p / s

                print(f"\n>>> [Simulation Matrix {current_run_idx}/{total_runs}] Starting: Dataset={d} | Partition={p} | Strategy={s.upper()}")

                metrics = run_single_experiment(
                    strategy=s,
                    partition_mode=p,
                    dataset_name=d,
                    partition_path=partition_file,
                    num_clients=num_clients,
                    rounds=rounds,
                    local_epochs=local_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    lr_mode=lr_mode,
                    client_lr_patience=client_lr_patience,
                    client_lr_factor=client_lr_factor,
                    client_lr_min=client_lr_min,
                    lr_decay_steps=lr_decay_steps,
                    lr_decay_gamma=lr_decay_gamma,
                    proximal_mu=proximal_mu,
                    early_stop_patience=early_stop_patience,
                    early_stop_metric=early_stop_metric,
                    server_eval=server_eval,
                    device_type=device_type,
                    save_client_metrics=save_client_metrics,
                    timeout_per_run=timeout_per_run,
                    run_dir=run_output_dir,
                    config_path=args.config,
                    num_workers=num_workers,
                    client_cpu_threads=client_cpu_threads,
                )

                summary_row = {k: v for k, v in metrics.items() if k != "rounds_history"}
                results_list.append(summary_row)

                for rh in metrics.get("rounds_history", []):
                    rounds_history_list.append({
                        "dataset": d,
                        "partition_mode": p,
                        "strategy": s,
                        "round": rh.get("round", 0),
                        "eval_accuracy": rh.get("eval_accuracy", 0.0),
                        "eval_loss": rh.get("eval_loss", 0.0),
                        "train_accuracy_avg": rh.get("train_accuracy_avg", 0.0),
                        "train_loss_avg": rh.get("train_loss_avg", 0.0),
                        "round_time_seconds": rh.get("round_time_seconds", 0.0),
                        "total_elapsed_seconds": rh.get("total_elapsed_seconds", 0.0),
                    })

                df_summary_temp = pd.DataFrame(results_list)
                df_summary_temp.to_csv(base_output_dir / "simulation_summary.csv", index=False)

    total_sim_time = time.time() - sim_start_time

    df_summary = pd.DataFrame(results_list)
    df_history = pd.DataFrame(rounds_history_list)

    summary_csv_path = base_output_dir / "simulation_summary.csv"
    history_csv_path = base_output_dir / "simulation_rounds_history.csv"
    df_summary.to_csv(summary_csv_path, index=False)
    df_history.to_csv(history_csv_path, index=False)

    print("\nGenerating multi-strategy comparison plots grouped by dataset...")
    if not df_history.empty and "dataset" in df_history.columns:
        for d in datasets:
            d_hist = df_history[df_history["dataset"] == d]
            d_summ = df_summary[df_summary["dataset"] == d] if (not df_summary.empty and "dataset" in df_summary.columns) else pd.DataFrame()
            if not d_hist.empty:
                plot_accuracy_curves(d_hist, d, partitions, base_output_dir / f"{d}_compare_accuracy_curves.png")
                plot_loss_curves(d_hist, d, partitions, base_output_dir / f"{d}_compare_loss_curves.png")
            if not d_summ.empty:
                plot_accuracy_barchart(d_summ, d, strategies, partitions, base_output_dir / f"{d}_compare_accuracy_barchart.png")
                plot_runtime_barchart(d_summ, d, strategies, partitions, base_output_dir / f"{d}_compare_runtime_barchart.png")
                plot_accuracy_heatmap(d_summ, d, strategies, partitions, base_output_dir / f"{d}_compare_accuracy_heatmap.png")
    else:
        print("  [Notice] No rounds history available to plot (runs may have timed out or failed before completing).")

    sim_meta = {
        "timestamp": timestamp_str,
        "config": cfg,
        "total_wall_clock_time_seconds": total_sim_time,
        "total_runs": total_runs,
        "completed_runs": len(results_list),
        "results": results_list,
    }
    with open(base_output_dir / "simulation_results.json", "w", encoding="utf-8") as f:
        json.dump(sim_meta, f, indent=4)

    report_path = base_output_dir / "simulation_report.md"
    generate_markdown_report(
        df_summary=df_summary,
        strategies=strategies,
        datasets=datasets,
        partitions=partitions,
        sim_config={
            "rounds": rounds,
            "num_clients": num_clients,
            "local_epochs": local_epochs,
            "lr_mode": lr_mode,
        },
        output_path=report_path,
    )

    print(f"\n{'='*75}")
    print(f"  FedMedAI Simulation Complete! Total Wall-Clock Time: {total_sim_time/60:.1f} minutes")
    print(f"{'='*75}")
    print(f"  Artifacts Saved in Directory: {base_output_dir}")
    print(f"  - Executive Report     : {report_path.name}")
    print(f"{'='*75}\n")

if __name__ == "__main__":
    main()
