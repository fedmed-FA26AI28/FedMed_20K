"""Multi-Strategy & Multi-Partition Simulation Runner for FedMedAI.

Executes a grid of federated learning experiments across multiple aggregation strategies
(FedAvg, FedProx, FedNova, FedBN, SCAFFOLD) and Dirichlet non-IID data partitions
(e.g., alpha = 1.0, 0.3, 0.1).

Features:
- Configurable via dedicated YAML file (`configs/simulation.yaml`) and CLI arguments.
- Automated Dirichlet partition verification and on-demand generation.
- Dynamic socket port management to avoid collisions between consecutive runs.
- Subprocess isolation mirroring real multi-device edge deployment.
- Real-time logging, graceful interrupt (Ctrl+C) handling, and error recovery.
- Produces individual run artifacts (CSVs, PNGs, checkpoints, JSON).
- Generates comprehensive cross-strategy comparison artifacts:
    * `simulation_summary.csv`
    * `simulation_rounds_history.csv`
    * `simulation_results.json`
    * `simulation_report.md`
    * `compare_accuracy_curves.png`
    * `compare_loss_curves.png`
    * `compare_accuracy_barchart.png`
    * `compare_runtime_barchart.png`
    * `compare_accuracy_heatmap.png`

Usage:
    # Run full simulation matrix from configs/simulation.yaml:
    python -m experiments.run_simulation

    # Custom config file:
    python -m experiments.run_simulation --config configs/simulation.yaml

    # Override parameters via CLI:
    python -m experiments.run_simulation --rounds 5 --strategies fedavg fedprox --alphas 0.3 0.1

    # Fast verification run (1 round, 2 clients, fedavg):
    python -m experiments.run_simulation --rounds 1 --num_clients 2 --strategies fedavg --alphas 1.0
"""

import argparse
import datetime
import json
import os
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

from datasets.medmnist_code import get_bloodmnist_datasets
from datasets.partition import dirichlet_partition, save_partition, visualize_partition


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
    num_clients: int,
    alpha: float,
    seed: int = 42,
    partition_dir: str = "data/partitions",
) -> Path:
    """Ensure Dirichlet partition JSON exists for (num_clients, alpha).

    If missing, automatically generates the partition using BloodMNIST training set
    and validates distribution integrity.
    """
    part_dir = Path(partition_dir)
    part_dir.mkdir(parents=True, exist_ok=True)
    part_file = part_dir / f"partition_seed{seed}_alpha{alpha}_clients{num_clients}.json"

    if part_file.exists():
        return part_file

    print(f"\n[Partition] Missing partition: {part_file.name}. Generating automatically...")
    train_dataset, _, _, _ = get_bloodmnist_datasets(download=True)
    partition = dirichlet_partition(train_dataset, num_clients, alpha, seed)

    # Validate integrity
    total_assigned = sum(len(indices) for indices in partition)
    assert total_assigned == len(train_dataset), "Partition sample count mismatch!"
    all_indices = [idx for sublist in partition for idx in sublist]
    assert len(all_indices) == len(set(all_indices)), "Found duplicate sample indices across clients!"

    # Save mapping and visualization
    save_partition(partition, train_dataset, alpha, seed, num_clients, save_dir=str(part_dir))
    plot_path = part_dir / "plots" / f"dist_alpha{alpha}_clients{num_clients}.png"
    visualize_partition(partition, train_dataset, alpha, num_clients, save_path=str(plot_path))
    print(f"[Partition] Generated and verified: {part_file} (plot: {plot_path})")

    return part_file


# ──────────────────────────────────────────────────────────────────────────────
# Single FL Experiment Runner (Subprocess Architecture)
# ──────────────────────────────────────────────────────────────────────────────

def run_single_experiment(
    strategy: str,
    alpha: float,
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
    print(f"  Launching FL Server: {strategy.upper()} | Alpha: {alpha} | Port: {port}")
    print(f"  Artifact Directory : {run_dir}")
    print(f"{'-'*65}")

    # Build server command
    server_cmd = [
        python_exe, "-m", "server.server",
        "--config", str(config_path),
        "--strategy", strategy,
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
    server_proc = subprocess.Popen(
        server_cmd,
        cwd=str(PROJECT_ROOT),
        stdout=server_log_file,
        stderr=subprocess.STDOUT,
    )

    client_procs = []
    client_log_files = []
    run_start_time = time.time()
    status = "failed"
    run_metrics: Dict[str, Any] = {}

    try:
        # Wait for server socket to become ready
        if not wait_for_server(host, port, timeout=25.0):
            raise TimeoutError(f"Server on {host}:{port} did not start listening within timeout.")

        print(f"  [Server] Listening. Launching {num_clients} client workers...")

        # Launch client subprocesses
        for cid in range(num_clients):
            c_dir = clients_dir / f"client_{cid}"
            c_dir.mkdir(parents=True, exist_ok=True)
            c_log_path = c_dir / "client.log"
            c_log_file = open(c_log_path, "w", encoding="utf-8")
            client_log_files.append(c_log_file)

            client_cmd = [
                python_exe, "-m", "client.client",
                "--config", str(config_path),
                "--client_id", str(cid),
                "--server_address", f"{host}:{port}",
                "--partition_path", str(partition_path),
                "--alpha", str(alpha),
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
            )
            client_procs.append(c_proc)

        # Active supervision loop: monitor server and all clients concurrently
        loop_start = time.time()
        while True:
            # Check if server completed
            server_exit = server_proc.poll()
            if server_exit is not None:
                if server_exit == 0:
                    status = "completed"
                else:
                    status = f"server_error({server_exit})"
                break

            # Check if any client crashed early with non-zero exit code
            failed_clients = []
            for c_idx, cp in enumerate(client_procs):
                c_exit = cp.poll()
                if c_exit is not None and c_exit != 0:
                    failed_clients.append((c_idx, c_exit))

            if failed_clients:
                cid, err_code = failed_clients[0]
                status = f"client_error({err_code})"
                print(f"  [Error] Client {cid} exited with error code {err_code}!")
                c_log_path = clients_dir / f"client_{cid}" / "client.log"
                if c_log_path.exists():
                    with open(c_log_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        print(f"  --- Tail of client_{cid} log ---")
                        print("  " + "".join(lines[-8:]).replace("\n", "\n  "))
                server_proc.kill()
                for cp in client_procs:
                    cp.kill()
                break

            # Check timeout
            if time.time() - loop_start > timeout_per_run:
                print(f"  [Timeout] Run exceeded timeout limit of {timeout_per_run}s.")
                server_proc.kill()
                for cp in client_procs:
                    cp.kill()
                status = "timeout"
                break

            time.sleep(0.5)

        # Wait for clients to finish cleanly
        for c_proc in client_procs:
            try:
                c_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                c_proc.kill()

    except subprocess.TimeoutExpired:
        print(f"  [Error] Run timed out after {timeout_per_run}s. Terminating processes...")
        status = "timeout"
        server_proc.kill()
        for c_proc in client_procs:
            c_proc.kill()
    except KeyboardInterrupt:
        print("\n  [Simulation] User interrupted run (Ctrl+C). Terminating subprocesses...")
        server_proc.kill()
        for c_proc in client_procs:
            c_proc.kill()
        raise
    except Exception as e:
        print(f"  [Error] Encountered exception during execution: {e}")
        status = f"error: {type(e).__name__}"
        server_proc.kill()
        for c_proc in client_procs:
            c_proc.kill()
    finally:
        server_log_file.close()
        for f in client_log_files:
            f.close()

    total_wall_time = time.time() - run_start_time

    # Parse and extract artifacts from fl_results.json
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
                "alpha": float(alpha),
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
            print(f"  [Success] Finished {strategy.upper()} (Alpha={alpha}): "
                  f"Best Acc: {run_metrics['best_test_accuracy']*100:.2f}% (R{run_metrics['best_round']}) | "
                  f"Time: {run_metrics['total_time_seconds']:.1f}s")
            return run_metrics
        except Exception as e:
            print(f"  [Warning] Failed to parse fl_results.json: {e}")

    # Fallback if results file could not be parsed
    run_metrics = {
        "strategy": strategy,
        "alpha": float(alpha),
        "num_clients": num_clients,
        "rounds_requested": rounds,
        "rounds_completed": 0,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "lr_mode": lr_mode,
        "learning_rate": learning_rate,
        "final_test_accuracy": 0.0,
        "best_test_accuracy": 0.0,
        "best_round": 0,
        "final_test_loss": 0.0,
        "best_test_loss": 0.0,
        "total_time_seconds": total_wall_time,
        "avg_round_time_seconds": 0.0,
        "early_stopped": False,
        "status": status,
        "run_dir": str(run_dir),
        "rounds_history": [],
    }
    return run_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Visualization & Comparison Artifact Generators
# ──────────────────────────────────────────────────────────────────────────────

STRATEGY_COLORS = {
    "fedavg": "#2563eb",   # Vibrant Blue
    "fedprox": "#16a34a",  # Fresh Green
    "fednova": "#ea580c",  # Warm Orange
    "fedbn": "#9333ea",    # Purple
    "scaffold": "#dc2626", # Deep Red
    "proposed": "#0891b2", # Cyan / Teal
}

STRATEGY_MARKERS = {
    "fedavg": "o",
    "fedprox": "s",
    "fednova": "^",
    "fedbn": "D",
    "scaffold": "v",
    "proposed": "P",
}


def plot_accuracy_curves(df_history: pd.DataFrame, alphas: List[float], output_path: Path):
    """Plot Test Accuracy vs Round curves for all strategies across alpha partitions."""
    if df_history.empty:
        return

    num_alphas = len(alphas)
    fig, axes = plt.subplots(1, num_alphas, figsize=(5.5 * num_alphas, 4.5), sharey=True, squeeze=False)
    axes = axes[0]

    for idx, alpha in enumerate(alphas):
        ax = axes[idx]
        sub_df = df_history[df_history["alpha"] == alpha]

        for strat, strat_group in sub_df.groupby("strategy"):
            strat_group = strat_group.sort_values("round")
            color = STRATEGY_COLORS.get(strat.lower(), "#64748b")
            marker = STRATEGY_MARKERS.get(strat.lower(), "o")

            # Convert to percentage
            acc_pct = strat_group["eval_accuracy"] * 100
            ax.plot(
                strat_group["round"],
                acc_pct,
                label=strat.upper(),
                color=color,
                marker=marker,
                markersize=4.5,
                linewidth=1.8,
                alpha=0.9,
            )

        degree_str = "Near-IID" if alpha >= 1.0 else ("Moderate Non-IID" if alpha >= 0.3 else "High Non-IID")
        ax.set_title(f"Alpha = {alpha} ({degree_str})", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Federated Round", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Global Test Accuracy (%)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=8.5, loc="lower right")

    fig.suptitle("FedMedAI Simulation: Global Test Accuracy Convergence Across Strategies", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(df_history: pd.DataFrame, alphas: List[float], output_path: Path):
    """Plot Global Test Loss vs Round curves for all strategies across alpha partitions."""
    if df_history.empty:
        return

    num_alphas = len(alphas)
    fig, axes = plt.subplots(1, num_alphas, figsize=(5.5 * num_alphas, 4.5), sharey=True, squeeze=False)
    axes = axes[0]

    for idx, alpha in enumerate(alphas):
        ax = axes[idx]
        sub_df = df_history[df_history["alpha"] == alpha]

        for strat, strat_group in sub_df.groupby("strategy"):
            strat_group = strat_group.sort_values("round")
            color = STRATEGY_COLORS.get(strat.lower(), "#64748b")
            marker = STRATEGY_MARKERS.get(strat.lower(), "o")

            ax.plot(
                strat_group["round"],
                strat_group["eval_loss"],
                label=strat.upper(),
                color=color,
                marker=marker,
                markersize=4.5,
                linewidth=1.8,
                alpha=0.9,
            )

        degree_str = "Near-IID" if alpha >= 1.0 else ("Moderate Non-IID" if alpha >= 0.3 else "High Non-IID")
        ax.set_title(f"Alpha = {alpha} ({degree_str})", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Federated Round", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Global Test Loss", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=8.5, loc="upper right")

    fig.suptitle("FedMedAI Simulation: Global Test Loss Convergence Across Strategies", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_barchart(df_summary: pd.DataFrame, strategies: List[str], alphas: List[float], output_path: Path):
    """Generate grouped bar chart comparing Best Global Test Accuracy across (Strategy x Alpha)."""
    if df_summary.empty:
        return

    pivot_acc = df_summary.pivot(index="alpha", columns="strategy", values="best_test_accuracy") * 100
    # Reindex columns and rows if present
    existing_strats = [s for s in strategies if s in pivot_acc.columns]
    existing_alphas = [a for a in alphas if a in pivot_acc.index]
    pivot_acc = pivot_acc.reindex(index=existing_alphas, columns=existing_strats)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(existing_alphas))
    width = 0.8 / max(len(existing_strats), 1)

    for i, strat in enumerate(existing_strats):
        vals = pivot_acc[strat].values
        color = STRATEGY_COLORS.get(strat.lower(), "#64748b")
        rects = ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=strat.upper(), color=color, alpha=0.9, edgecolor="black", linewidth=0.5)

        # Value labels above bars
        for rect in rects:
            h = rect.get_height()
            if not np.isnan(h) and h > 0:
                ax.annotate(
                    f"{h:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    fontweight="bold",
                )

    ax.set_ylabel("Best Test Accuracy (%)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Dirichlet Concentration Alpha (Non-IID Degree)", fontsize=10, fontweight="bold")
    ax.set_title("Cross-Strategy Performance Comparison Across Data Heterogeneity", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Alpha = {a}\n({'Near-IID' if a>=1.0 else ('Moderate' if a>=0.3 else 'High Non-IID')})" for a in existing_alphas], fontsize=9)
    ax.set_ylim(0, min(100, max(pivot_acc.max().max() * 1.15, 50) if not pivot_acc.empty else 100))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_runtime_barchart(df_summary: pd.DataFrame, strategies: List[str], alphas: List[float], output_path: Path):
    """Generate grouped bar chart comparing Total Execution Time (seconds) across (Strategy x Alpha)."""
    if df_summary.empty:
        return

    pivot_time = df_summary.pivot(index="alpha", columns="strategy", values="total_time_seconds")
    existing_strats = [s for s in strategies if s in pivot_time.columns]
    existing_alphas = [a for a in alphas if a in pivot_time.index]
    pivot_time = pivot_time.reindex(index=existing_alphas, columns=existing_strats)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(existing_alphas))
    width = 0.8 / max(len(existing_strats), 1)

    for i, strat in enumerate(existing_strats):
        vals = pivot_time[strat].values
        color = STRATEGY_COLORS.get(strat.lower(), "#64748b")
        rects = ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=strat.upper(), color=color, alpha=0.9, edgecolor="black", linewidth=0.5)

        for rect in rects:
            h = rect.get_height()
            if not np.isnan(h) and h > 0:
                ax.annotate(
                    f"{h:.0f}s",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )

    ax.set_ylabel("Total Wall-Clock Time (seconds)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Dirichlet Concentration Alpha", fontsize=10, fontweight="bold")
    ax.set_title("Computational & Wall-Clock Runtime Comparison Across Strategies", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Alpha = {a}" for a in existing_alphas], fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_heatmap(df_summary: pd.DataFrame, strategies: List[str], alphas: List[float], output_path: Path):
    """Generate 2D heatmap matrix (Strategy x Alpha) showing Best Test Accuracy."""
    if df_summary.empty:
        return

    pivot_acc = df_summary.pivot(index="strategy", columns="alpha", values="best_test_accuracy") * 100
    existing_strats = [s for s in strategies if s in pivot_acc.index]
    existing_alphas = [a for a in alphas if a in pivot_acc.columns]
    pivot_acc = pivot_acc.reindex(index=existing_strats, columns=existing_alphas)

    fig, ax = plt.subplots(figsize=(6.5, max(4.0, len(existing_strats) * 0.8)))
    data = pivot_acc.values

    im = ax.imshow(data, cmap="YlGnBu", aspect="auto")

    # Show values in cells
    for i in range(len(existing_strats)):
        for j in range(len(existing_alphas)):
            val = data[i, j]
            if not np.isnan(val):
                text_color = "white" if val > (np.nanmax(data) + np.nanmin(data)) / 2 else "black"
                ax.text(j, i, f"{val:.2f}%", ha="center", va="center", color=text_color, fontweight="bold", fontsize=10)

    ax.set_xticks(np.arange(len(existing_alphas)))
    ax.set_yticks(np.arange(len(existing_strats)))
    ax.set_xticklabels([f"Alpha = {a}" for a in existing_alphas], fontsize=10, fontweight="bold")
    ax.set_yticklabels([s.upper() for s in existing_strats], fontsize=10, fontweight="bold")
    ax.set_title("Best Test Accuracy Heatmap Matrix (Strategy x Alpha)", fontsize=11, fontweight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Accuracy (%)", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_markdown_report(
    df_summary: pd.DataFrame,
    strategies: List[str],
    alphas: List[float],
    sim_config: Dict[str, Any],
    output_path: Path,
):
    """Generate an executive Markdown summary report with comparative leaderboards."""
    lines = [
        "# FedMedAI Simulation Benchmark Report",
        "",
        f"**Generated At:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Rounds:** {sim_config.get('rounds')} | **Clients:** {sim_config.get('num_clients')} | "
        f"**Local Epochs:** {sim_config.get('local_epochs')} | **LR Mode:** `{sim_config.get('lr_mode')}`  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Best Performing Strategies",
        "",
    ]

    # Find winner per alpha
    for alpha in alphas:
        sub = df_summary[df_summary["alpha"] == alpha]
        if not sub.empty:
            best_row = sub.loc[sub["best_test_accuracy"].idxmax()]
            degree_str = "Near-IID" if alpha >= 1.0 else ("Moderate Non-IID" if alpha >= 0.3 else "High Non-IID")
            lines.append(
                f"- **Alpha = {alpha} ({degree_str}):** 🏆 **{best_row['strategy'].upper()}** achieved "
                f"**{best_row['best_test_accuracy']*100:.2f}%** accuracy in Round {int(best_row['best_round'])} "
                f"(total runtime: {best_row['total_time_seconds']:.1f}s)."
            )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Full Simulation Results Matrix",
        "",
        "| Strategy | Alpha | Final Acc (%) | Best Acc (%) | Best Round | Final Loss | Best Loss | Total Time | Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for _, row in df_summary.iterrows():
        lines.append(
            f"| **{row['strategy'].upper()}** | `{row['alpha']}` | "
            f"{row['final_test_accuracy']*100:.2f}% | **{row['best_test_accuracy']*100:.2f}%** | "
            f"Round {int(row['best_round'])} | {row['final_test_loss']:.4f} | {row['best_test_loss']:.4f} | "
            f"{row['total_time_seconds']:.1f}s | `{row['status']}` |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Generated Comparison Artifacts",
        "",
        "- 📊 **Accuracy Curves:** `compare_accuracy_curves.png`",
        "- 📉 **Loss Curves:** `compare_loss_curves.png`",
        "- 📶 **Accuracy Bar Chart:** `compare_accuracy_barchart.png`",
        "- ⏱️ **Runtime Comparison:** `compare_runtime_barchart.png`",
        "- 🗺️ **Accuracy Heatmap:** `compare_accuracy_heatmap.png`",
        "- 📋 **Aggregated Summary CSV:** `simulation_summary.csv`",
        "- 📜 **Round History CSV:** `simulation_rounds_history.csv`",
        "- 💾 **Machine-Readable JSON:** `simulation_results.json`",
        "",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# Main Simulation Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FedMedAI Automated Multi-Strategy & Multi-Partition Simulation Benchmark"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/simulation.yaml",
        help="Path to YAML simulation configuration file (default: configs/simulation.yaml)",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=None,
        help="Number of simulated clients (default: from config, e.g. 3)",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=None,
        help="List of Dirichlet alpha values to benchmark (default: [1.0, 0.3, 0.1])",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=None,
        help="List of FL strategies to benchmark (default: ['fedavg', 'fedprox', 'fednova', 'fedbn', 'scaffold'])",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Number of FL communication rounds per run (default: from config)",
    )
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=None,
        help="Number of client local epochs per round (default: from config)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Client training mini-batch size (default: from config)",
    )
    parser.add_argument(
        "--learning_rate", "--lr",
        type=float,
        default=None,
        help="Initial learning rate (default: from config)",
    )
    parser.add_argument(
        "--lr_mode",
        type=str,
        choices=["client_loss", "server_decay", "fixed"],
        default=None,
        help="LR scheduling mode (default: from config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Base output directory to store simulation results (default: results/simulation/<timestamp>)",
    )
    parser.add_argument(
        "--timeout_per_run",
        type=int,
        default=None,
        help="Timeout in seconds for a single FL run (default: 1800)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="DataLoader worker processes per client (default: from config, 0 on Windows)",
    )
    parser.add_argument(
        "--server_eval",
        dest="server_eval",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable centralized evaluation on server test set (default: True)",
    )

    args = parser.parse_args()

    # Load YAML base configuration
    cfg = load_yaml_config(args.config)

    # Resolve parameters: CLI arguments override YAML settings
    num_clients = args.num_clients if args.num_clients is not None else cfg.get("num_clients", 3)
    alphas = args.alphas if args.alphas is not None else cfg.get("alphas", [1.0, 0.3, 0.1])
    strategies = args.strategies if args.strategies is not None else cfg.get("strategies", ["fedavg", "fedprox", "fednova", "fedbn", "scaffold"])
    rounds = args.rounds if args.rounds is not None else cfg.get("rounds", 10)
    local_epochs = args.local_epochs if args.local_epochs is not None else cfg.get("local_epochs", 3)
    batch_size = args.batch_size if args.batch_size is not None else cfg.get("batch_size", 32)
    num_workers = args.num_workers if args.num_workers is not None else cfg.get("num_workers", 0 if sys.platform == "win32" else 0)
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
    timeout_per_run = args.timeout_per_run if args.timeout_per_run is not None else cfg.get("timeout_per_run", 1800)
    seed = cfg.get("seed", 42)

    # Normalize types
    alphas = [float(a) for a in alphas]
    strategies = [str(s).lower().strip() for s in strategies]

    # Setup output directory
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        base_output_dir = Path(cfg.get("output_dir", "results/simulation")) / timestamp_str
    base_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  FedMedAI Automated Simulation Benchmark")
    print(f"{'='*70}")
    print(f"  Config File      : {args.config}")
    print(f"  Output Directory : {base_output_dir}")
    print(f"  Clients          : {num_clients}")
    print(f"  Rounds per Run   : {rounds}")
    print(f"  Local Epochs     : {local_epochs}")
    print(f"  Batch Size       : {batch_size}")
    print(f"  DataLoader Workers : {num_workers}")
    print(f"  LR Mode          : {lr_mode.upper()} (Initial LR: {learning_rate})")
    print(f"  Dirichlet Alphas : {alphas}")
    print(f"  Strategies       : {[s.upper() for s in strategies]}")
    total_runs = len(alphas) * len(strategies)
    print(f"  Total Runs Grid  : {total_runs} combinations ({len(alphas)} alphas x {len(strategies)} strategies)")
    print(f"{'='*70}\n")

    # Step 1: Ensure all required Dirichlet partitions exist
    print("Verifying data partition files for all alpha levels...")
    partition_map: Dict[float, Path] = {}
    for alpha in alphas:
        p_path = ensure_partition_exists(num_clients=num_clients, alpha=alpha, seed=seed)
        partition_map[alpha] = p_path

    # Step 2: Iterate through matrix grid (Alpha x Strategy)
    results_list: List[Dict[str, Any]] = []
    rounds_history_list: List[Dict[str, Any]] = []
    current_run_idx = 0

    sim_start_time = time.time()

    for alpha in alphas:
        partition_file = partition_map[alpha]

        for strategy in strategies:
            current_run_idx += 1
            run_tag = f"alpha_{alpha}/{strategy}"
            run_output_dir = base_output_dir / f"alpha_{alpha}" / strategy

            print(f"\n>>> [Simulation Matrix {current_run_idx}/{total_runs}] Starting: Strategy={strategy.upper()} | Alpha={alpha}")

            metrics = run_single_experiment(
                strategy=strategy,
                alpha=alpha,
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
            )

            # Record summary metric row
            summary_row = {k: v for k, v in metrics.items() if k != "rounds_history"}
            results_list.append(summary_row)

            # Record round-by-round history
            for rh in metrics.get("rounds_history", []):
                rounds_history_list.append({
                    "strategy": strategy,
                    "alpha": alpha,
                    "round": rh.get("round", 0),
                    "eval_accuracy": rh.get("eval_accuracy", 0.0),
                    "eval_loss": rh.get("eval_loss", 0.0),
                    "train_accuracy_avg": rh.get("train_accuracy_avg", 0.0),
                    "train_loss_avg": rh.get("train_loss_avg", 0.0),
                    "round_time_seconds": rh.get("round_time_seconds", 0.0),
                    "total_elapsed_seconds": rh.get("total_elapsed_seconds", 0.0),
                })

            # Intermediate save after each run
            df_summary_temp = pd.DataFrame(results_list)
            df_summary_temp.to_csv(base_output_dir / "simulation_summary.csv", index=False)

    total_sim_time = time.time() - sim_start_time

    # Step 3: Build master DataFrames & CSVs
    df_summary = pd.DataFrame(results_list)
    df_history = pd.DataFrame(rounds_history_list)

    summary_csv_path = base_output_dir / "simulation_summary.csv"
    history_csv_path = base_output_dir / "simulation_rounds_history.csv"
    df_summary.to_csv(summary_csv_path, index=False)
    df_history.to_csv(history_csv_path, index=False)

    # Step 4: Generate Publication-Grade Comparison Visualizations
    print("\nGenerating multi-strategy cross-partition comparison plots...")
    plot_accuracy_curves(df_history, alphas, base_output_dir / "compare_accuracy_curves.png")
    plot_loss_curves(df_history, alphas, base_output_dir / "compare_loss_curves.png")
    plot_accuracy_barchart(df_summary, strategies, alphas, base_output_dir / "compare_accuracy_barchart.png")
    plot_runtime_barchart(df_summary, strategies, alphas, base_output_dir / "compare_runtime_barchart.png")
    plot_accuracy_heatmap(df_summary, strategies, alphas, base_output_dir / "compare_accuracy_heatmap.png")

    # Step 5: Save JSON Benchmark Object and Markdown Report
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
        alphas=alphas,
        sim_config={
            "rounds": rounds,
            "num_clients": num_clients,
            "local_epochs": local_epochs,
            "lr_mode": lr_mode,
        },
        output_path=report_path,
    )

    # Step 6: Print Console Summary Table
    print(f"\n{'='*75}")
    print(f"  FedMedAI Simulation Complete! Total Wall-Clock Time: {total_sim_time/60:.1f} minutes")
    print(f"{'='*75}")
    print(f"  Artifacts Saved in Directory: {base_output_dir}")
    print(f"  - Summary Table CSV    : {summary_csv_path.name}")
    print(f"  - Round History CSV    : {history_csv_path.name}")
    print(f"  - Comparison Curves    : compare_accuracy_curves.png & compare_loss_curves.png")
    print(f"  - Grouped Bar Charts   : compare_accuracy_barchart.png & compare_runtime_barchart.png")
    print(f"  - Heatmap Matrix       : compare_accuracy_heatmap.png")
    print(f"  - Executive Report     : {report_path.name}")
    print(f"{'='*75}\n")

    # Print formatted console table
    display_cols = ["strategy", "alpha", "best_test_accuracy", "best_round", "final_test_loss", "total_time_seconds", "status"]
    print(df_summary[display_cols].to_string(index=False))
    print(f"\nDone! Benchmark finished successfully.")


if __name__ == "__main__":
    main()
