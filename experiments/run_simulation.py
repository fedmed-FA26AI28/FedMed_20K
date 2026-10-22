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
from datasets.partition import (
    dirichlet_partition,
    pathological_partition,
    practical_partition,
    save_partition,
    visualize_partition,
    has_local_test,
    get_partition_path,
    get_partition_plot_path,
)
from utils.path_utils import sanitize_run_name


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
    local_test: bool = False,
) -> Path:
    """Ensure partition JSON exists for (dataset_name, partition_mode, num_clients).

    If missing or lacking local_test when local_test is requested, automatically generates the partition.
    """
    part_dir = Path(partition_dir)
    part_dir.mkdir(parents=True, exist_ok=True)
    part_file = get_partition_path(
        dataset_name=dataset_name,
        partition_mode=partition_mode,
        seed=seed,
        num_clients=num_clients,
        local_test=local_test,
        save_dir=part_dir,
    )

    if part_file.exists():
        if not local_test or has_local_test(part_file):
            return part_file
        print(f"\n[Partition] Existing partition {part_file.name} lacks local test partition. Regenerating with local_test=True...")
    else:
        print(f"\n[Partition] Missing partition: {part_file.name}. Generating automatically (local_test={local_test})...")

    train_dataset, _, test_dataset, _ = get_medmnist_datasets(dataset_name=dataset_name, download=True)
    target_test_dataset = test_dataset if local_test else None
    
    if partition_mode == "pathological":
        res = pathological_partition(train_dataset, num_clients, seed, test_dataset=target_test_dataset)
    elif partition_mode == "practical":
        res = practical_partition(train_dataset, num_clients, seed, test_dataset=target_test_dataset)
    elif partition_mode.startswith("dirichlet_"):
        alpha = float(partition_mode.split("_")[1])
        res = dirichlet_partition(train_dataset, num_clients, alpha, seed, test_dataset=target_test_dataset)
    else:
        raise ValueError(f"Unknown partition mode: {partition_mode}")

    if local_test:
        partition, test_partition = res
    else:
        partition = res
        test_partition = None

    # Validate integrity
    total_assigned = sum(len(indices) for indices in partition)
    assert total_assigned == len(train_dataset), "Partition train sample count mismatch!"
    all_indices = [idx for sublist in partition for idx in sublist]
    assert len(all_indices) == len(set(all_indices)), "Found duplicate sample indices across clients in train!"

    if local_test:
        total_test_assigned = sum(len(indices) for indices in test_partition)
        assert total_test_assigned == len(test_dataset), "Partition test sample count mismatch!"
        all_test_indices = [idx for sublist in test_partition for idx in sublist]
        assert len(all_test_indices) == len(set(all_test_indices)), "Found duplicate sample indices across clients in test!"

    # Save mapping and visualization
    saved_path = save_partition(
        partition,
        train_dataset,
        dataset_name,
        partition_mode,
        seed,
        num_clients,
        test_dataset=test_dataset if local_test else None,
        test_indices=test_partition if local_test else None,
        save_dir=str(part_dir),
    )
    plot_path = get_partition_plot_path(
        dataset_name=dataset_name,
        partition_mode=partition_mode,
        seed=seed,
        num_clients=num_clients,
        local_test=local_test,
        save_dir=part_dir,
    )
    try:
        visualize_partition(
            partition,
            train_dataset,
            dataset_name,
            partition_mode,
            num_clients,
            seed=seed,
            save_path=str(plot_path.resolve()),
            test_partition=test_partition if local_test else None,
            test_dataset=test_dataset if local_test else None,
        )
    except Exception as e:
        print(f"[Partition] Warning: Could not generate visualization plot: {e}")
    print(f"[Partition] Generated and verified: {saved_path} (plot: {plot_path})")

    return saved_path


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
    server_lr_min: float,
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
    local_test: bool = False,
    model: str = "cnn",
    seed: int = 42,
    optimizer: str = "adam",
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    augment: bool = True,
    random_crop: bool = True,
    random_flip: bool = True,
    client_module: str = "experiments.simulation_client",
) -> Dict[str, Any]:
    """Execute a single FL training run (1 Server + N Clients) as isolated subprocesses."""
    run_dir.mkdir(parents=True, exist_ok=True)
    server_log_path = run_dir / "server.log"
    clients_dir = run_dir / "clients"
    clients_dir.mkdir(parents=True, exist_ok=True)

    port = get_free_port()
    host = "127.0.0.1"

    print(f"\n{'-'*65}")
    print(f"  Launching FL Server: {strategy.upper()} | Model: {model.upper()} | Optimizer: {optimizer.upper()} | Dataset: {dataset_name} | Partition: {partition_mode} | Port: {port} | LocalTest: {local_test} | Seed: {seed}")
    print(f"  Artifact Directory : {run_dir}")
    print(f"{'-'*65}")

    # Build server command
    server_cmd = [
        python_exe, "-u", "-m", "server.server",
        "--config", str(config_path),
        "--strategy", strategy,
        "--dataset", dataset_name,
        "--model", model,
        "--optimizer", optimizer,
        "--momentum", str(momentum),
        "--weight_decay", str(weight_decay),
        "--seed", str(seed),
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
        "--server_lr_min", str(server_lr_min),
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
    if local_test:
        server_cmd.append("--local_test")
    else:
        server_cmd.append("--no-local_test")

    server_log_file = open(server_log_path, "w", encoding="utf-8")
    
    lock_path = run_dir / ".gpu.lock"
    if lock_path.exists():
        try:
            lock_path.unlink()
        except Exception:
            pass

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["FEDMEDAI_SIMULATION_LOCK_PATH"] = str(lock_path.resolve())
    if client_cpu_threads > 0:
        env["OMP_NUM_THREADS"] = str(client_cpu_threads)
        env["MKL_NUM_THREADS"] = str(client_cpu_threads)
        env["OPENBLAS_NUM_THREADS"] = str(client_cpu_threads)
        env["VECLIB_MAXIMUM_THREADS"] = str(client_cpu_threads)
        env["NUMEXPR_NUM_THREADS"] = str(client_cpu_threads)

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
        if not wait_for_server(host, port, timeout=180.0):
            raise TimeoutError(f"Server on {host}:{port} did not start listening within timeout.")

        print(f"  [Server] Listening. Launching {num_clients} client workers (Module: {client_module})...")

        for cid in range(num_clients):
            c_dir = clients_dir / f"client_{cid}"
            c_dir.mkdir(parents=True, exist_ok=True)
            c_log_path = c_dir / "client.log"
            c_log_file = open(c_log_path, "w", encoding="utf-8")
            client_log_files.append(c_log_file)

            client_cmd = [
                python_exe, "-u", "-m", client_module,
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
                "--model", model,
                "--optimizer", optimizer,
                "--momentum", str(momentum),
                "--weight_decay", str(weight_decay),
                "--seed", str(seed),
            ]
            if local_test:
                client_cmd.append("--local_test")
            else:
                client_cmd.append("--no-local_test")
            if augment:
                client_cmd.append("--augment")
            else:
                client_cmd.append("--no-augment")
            if random_crop:
                client_cmd.append("--random_crop")
            else:
                client_cmd.append("--no-random_crop")
            if random_flip:
                client_cmd.append("--random_flip")
            else:
                client_cmd.append("--no-random_flip")
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
        if lock_path.exists():
            try:
                lock_path.unlink()
            except Exception:
                pass

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
                "model": model,
                "optimizer": optimizer,
                "momentum": momentum,
                "seed": seed,
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
                "bmcta": float(summary["bmcta"]) if summary.get("bmcta") is not None else None,
                "bmcta_round": int(summary["bmcta_round"]) if summary.get("bmcta_round") is not None else None,
                "bta": float(summary["bta"]) if summary.get("bta") is not None else None,
                "bta_round": int(summary["bta_round"]) if summary.get("bta_round") is not None else None,
                "local_test": local_test,
                "augment": augment,
                "random_crop": random_crop,
                "random_flip": random_flip,
                "total_time_seconds": float(summary.get("total_wall_clock_time_seconds", total_wall_time)),
                "avg_round_time_seconds": float(summary.get("avg_round_time_seconds", 0.0)),
                "early_stopped": bool(summary.get("early_stopped", False) or metadata.get("early_stopped", False)),
                "status": status,
                "run_dir": str(run_dir),
                "rounds_history": rounds_history,
            }
            if local_test and run_metrics.get("bmcta") is not None:
                bmcta_str = f"BMCTA: {run_metrics['bmcta']*100:.2f}% (R{run_metrics.get('bmcta_round', 0)}) | "
                bta_str = f"BTA: {run_metrics['bta']*100:.2f}% (R{run_metrics.get('bta_round', 0)}) | " if run_metrics.get("bta") is not None else ""
                print(f"  [Success] Finished {strategy.upper()} ({partition_mode}) [Local Test]: "
                      f"{bmcta_str}{bta_str}"
                      f"Time: {run_metrics['total_time_seconds']:.1f}s")
            else:
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
        "model": model,
        "optimizer": optimizer,
        "momentum": momentum,
        "seed": seed,
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
    "fedbn_scaffold": "#0891b2",
    "bn_scaffold": "#0891b2",
    "scaffold_fedbn": "#0891b2",
    "proposed": "#0891b2",
}

STRATEGY_MARKERS = {
    "fedavg": "o",
    "fedprox": "s",
    "fednova": "^",
    "fedbn": "D",
    "scaffold": "v",
    "fedbn_scaffold": "P",
    "bn_scaffold": "P",
    "scaffold_fedbn": "P",
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
        (f"**Run Name:** `{sim_config['name']}` | " if sim_config.get("name") else "") +
        f"**Model:** `{sim_config.get('model', 'cnn')}` | **Seed:** `{sim_config.get('seed', 42)}` | "
        f"**Rounds:** {sim_config.get('rounds')} | **Clients:** {sim_config.get('num_clients')} | "
        f"**Local Epochs:** {sim_config.get('local_epochs')} | **LR Mode:** `{sim_config.get('lr_mode')}` | "
        f"**Local Test:** `{'Enabled' if sim_config.get('local_test') else 'Disabled'}`  ",
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
                lines.append(f"- **Partition = {p}:** **[BEST] {best_row['strategy'].upper()}** achieved **{best_row['best_test_accuracy']*100:.2f}%** accuracy in Round {int(best_row['best_round'])} (runtime: {best_row['total_time_seconds']:.1f}s).")
        lines.append("")

    has_bmcta = "bmcta" in df_summary.columns and df_summary["bmcta"].notna().any()
    has_model = "model" in df_summary.columns
    has_opt = "optimizer" in df_summary.columns
    has_aug = "augment" in df_summary.columns

    model_hdr = " Model |" if has_model else ""
    model_sep = " :---: |" if has_model else ""
    opt_hdr = " Opt |" if has_opt else ""
    opt_sep = " :---: |" if has_opt else ""
    aug_hdr = " Aug |" if has_aug else ""
    aug_sep = " :---: |" if has_aug else ""

    if has_bmcta:
        header = f"| Dataset |{model_hdr}{opt_hdr}{aug_hdr} Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |"
        sep = f"| :--- |{model_sep}{opt_sep}{aug_sep} :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        lines.extend([
            "---",
            "## 2. Full Simulation Results Matrix",
            "",
            header,
            sep,
        ])
        for _, row in df_summary.iterrows():
            bmcta_val = f"**{row['bmcta']*100:.2f}%** (R{int(row.get('bmcta_round', 0))})" if pd.notna(row.get("bmcta")) else "N/A"
            bta_val = f"**{row['bta']*100:.2f}%** (R{int(row.get('bta_round', 0))})" if pd.notna(row.get("bta")) else "N/A"
            model_col = f"`{row['model']}` | " if has_model else ""
            opt_col = f"`{row['optimizer']}` | " if has_opt else ""
            aug_col = f"{'Yes' if row.get('augment', True) else 'No'} | " if has_aug else ""
            lines.append(f"| **{row['dataset']}** | {model_col}{opt_col}{aug_col}**{row['strategy'].upper()}** | `{row['partition_mode']}` | {row['final_test_accuracy']*100:.2f}% | **{row['best_test_accuracy']*100:.2f}%** | {bmcta_val} | {bta_val} | R{int(row['best_round'])} | {row['final_test_loss']:.4f} | {row['total_time_seconds']:.1f}s |")
    else:
        header = f"| Dataset |{model_hdr}{opt_hdr}{aug_hdr} Strategy | Partition | Final Acc (%) | Best Acc (%) | Best Round | Final Loss | Total Time |"
        sep = f"| :--- |{model_sep}{opt_sep}{aug_sep} :--- | :--- | :---: | :---: | :---: | :---: | :---: |"
        lines.extend([
            "---",
            "## 2. Full Simulation Results Matrix",
            "",
            header,
            sep,
        ])
        for _, row in df_summary.iterrows():
            model_col = f"`{row['model']}` | " if has_model else ""
            opt_col = f"`{row['optimizer']}` | " if has_opt else ""
            aug_col = f"{'Yes' if row.get('augment', True) else 'No'} | " if has_aug else ""
            lines.append(f"| **{row['dataset']}** | {model_col}{opt_col}{aug_col}**{row['strategy'].upper()}** | `{row['partition_mode']}` | {row['final_test_accuracy']*100:.2f}% | **{row['best_test_accuracy']*100:.2f}%** | R{int(row['best_round'])} | {row['final_test_loss']:.4f} | {row['total_time_seconds']:.1f}s |")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# Main Simulation Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FedMedAI Automated Multi-Strategy & Multi-Partition Simulation")
    parser.add_argument("--config", type=str, default="configs/simulation.yaml")
    parser.add_argument("--num_clients", type=int, default=None)
    parser.add_argument("--datasets", type=str, nargs="+", default=None, help="Datasets to benchmark (e.g. bloodmnist pathmnist organamnist dermamnist)")
    parser.add_argument("--partitions", type=str, nargs="+", default=None)
    parser.add_argument("--strategies", type=str, nargs="+", default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--local_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", "--lr", type=float, default=None)
    parser.add_argument("--lr_mode", type=str, choices=["client_loss", "server_decay", "fixed"], default=None)
    parser.add_argument("--name", "-n", type=str, default=None, help="Designated simulation run name for output directory (default: timestamp)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--timeout_per_run", type=int, default=None, help="Max run timeout in seconds (<=0 to disable)")
    parser.add_argument("--no_timeout", dest="no_timeout", action="store_true", help="Disable run timeout completely")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--client_cpu_threads", type=int, default=None)
    parser.add_argument("--server_eval", dest="server_eval", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--local_test", dest="local_test", action=argparse.BooleanOptionalAction, default=None, help="Partition test set across clients and track BMCTA/BTA")
    parser.add_argument("--model", type=str, default=None, choices=["cnn", "baseline", "paper", "classic_cnn", "baseline_bn", "baselinecnn_bn", "classic_bn"], help="Model architecture ('cnn', 'baseline', or 'baseline_bn')")
    parser.add_argument("--optimizer", type=str, default=None, choices=["adam", "sgd"], help="Optimizer type ('adam' or 'sgd')")
    parser.add_argument("--momentum", type=float, default=None, help="Momentum for SGD optimizer")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay penalty")
    parser.add_argument("--seed", type=int, default=None, help="Global random seed")
    parser.add_argument("--augment", dest="augment", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable training data augmentation")
    parser.add_argument("--random_crop", dest="random_crop", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable random crop during training")
    parser.add_argument("--random_flip", dest="random_flip", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable random horizontal flip during training")
    parser.add_argument("--client_module", type=str, default=None, help="Python module to launch for clients (default: reads from config, fallback 'client.client')")
    parser.add_argument("--early_stop_patience", type=int, default=None, help="Patience rounds for early stopping (0 = disabled)")
    parser.add_argument("--early_stop_metric", type=str, choices=["accuracy", "loss"], default=None, help="Metric to monitor for early stopping")
    parser.add_argument("--lr_decay_steps", type=int, default=None, help="Communication rounds between step decays in server_decay mode")
    parser.add_argument("--lr_decay_gamma", type=float, default=None, help="Multiplicative decay factor in server_decay mode")
    parser.add_argument("--server_lr_min", "--lr_decay_min", type=float, default=None, help="Minimum learning rate floor for server_decay mode")
    parser.add_argument("--client_lr_patience", type=int, default=None, help="Local epochs without improvement before reducing client LR")
    parser.add_argument("--client_lr_factor", type=float, default=None, help="Factor for reducing client LR on plateau")
    parser.add_argument("--client_lr_min", type=float, default=None, help="Minimum client learning rate bound")
    parser.add_argument("--proximal_mu", type=float, default=None, help="Proximal coefficient for FedProx")

    args = parser.parse_args()
    cfg = load_yaml_config(args.config)

    num_clients = args.num_clients if args.num_clients is not None else cfg.get("num_clients", cfg.get("min_clients", 12))

    raw_datasets = args.datasets if args.datasets is not None else cfg.get("datasets", cfg.get("dataset", ["bloodmnist"]))
    if isinstance(raw_datasets, str):
        datasets = [raw_datasets]
    else:
        datasets = list(raw_datasets)

    raw_partitions = args.partitions if args.partitions is not None else cfg.get("partitions", cfg.get("partition", cfg.get("partition_mode", ["dirichlet_0.3", "pathological", "practical"])))
    if isinstance(raw_partitions, str):
        partitions = [raw_partitions]
    else:
        partitions = list(raw_partitions)

    raw_strategies = args.strategies if args.strategies is not None else cfg.get("strategies", cfg.get("strategy", ["fedavg", "fedprox", "fednova", "fedbn", "scaffold"]))
    if isinstance(raw_strategies, str):
        strategies = [raw_strategies]
    else:
        strategies = list(raw_strategies)

    rounds = args.rounds if args.rounds is not None else cfg.get("rounds", cfg.get("fl_rounds", 80))
    local_epochs = args.local_epochs if args.local_epochs is not None else cfg.get("local_epochs", 5)
    batch_size = args.batch_size if args.batch_size is not None else cfg.get("batch_size", 256)
    num_workers = args.num_workers if args.num_workers is not None else cfg.get("num_workers", 0 if sys.platform == "win32" else 0)
    client_cpu_threads = args.client_cpu_threads if args.client_cpu_threads is not None else cfg.get("client_cpu_threads", 1)
    learning_rate = args.learning_rate if args.learning_rate is not None else cfg.get("learning_rate", cfg.get("lr", 0.01))
    lr_mode = args.lr_mode if args.lr_mode is not None else cfg.get("lr_mode", "fixed")
    client_module = args.client_module if args.client_module is not None else cfg.get("client_module", "client.client")
    client_lr_patience = args.client_lr_patience if args.client_lr_patience is not None else cfg.get("client_lr_patience", 2)
    client_lr_factor = args.client_lr_factor if args.client_lr_factor is not None else cfg.get("client_lr_factor", 0.5)
    client_lr_min = args.client_lr_min if args.client_lr_min is not None else cfg.get("client_lr_min", 1e-6)
    lr_decay_steps = args.lr_decay_steps if args.lr_decay_steps is not None else cfg.get("lr_decay_steps", 3)
    lr_decay_gamma = args.lr_decay_gamma if args.lr_decay_gamma is not None else cfg.get("lr_decay_gamma", 0.5)
    server_lr_min = args.server_lr_min if args.server_lr_min is not None else cfg.get("server_lr_min", cfg.get("lr_decay_min", 1e-5))
    proximal_mu = args.proximal_mu if args.proximal_mu is not None else cfg.get("proximal_mu", 0.01)
    early_stop_patience = args.early_stop_patience if args.early_stop_patience is not None else cfg.get("early_stop_patience", 0)
    early_stop_metric = args.early_stop_metric if args.early_stop_metric is not None else cfg.get("early_stop_metric", "accuracy")
    server_eval = args.server_eval if args.server_eval is not None else cfg.get("server_eval", True)
    local_test = args.local_test if args.local_test is not None else cfg.get("local_test", False)
    device_type = cfg.get("device_type", "pc")
    save_client_metrics = cfg.get("save_client_metrics", True)
    model = args.model if args.model is not None else cfg.get("model", "baseline")
    optimizer = args.optimizer if args.optimizer is not None else cfg.get("optimizer", "sgd")
    momentum = args.momentum if args.momentum is not None else cfg.get("momentum", 0.0)
    weight_decay = args.weight_decay if args.weight_decay is not None else cfg.get("weight_decay", 0.0)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    augment = args.augment if args.augment is not None else cfg.get("augment", True)
    random_crop = args.random_crop if args.random_crop is not None else cfg.get("random_crop", True)
    random_flip = args.random_flip if args.random_flip is not None else cfg.get("random_flip", True)

    if getattr(args, "no_timeout", False):
        timeout_per_run = 0
    elif args.timeout_per_run is not None:
        timeout_per_run = args.timeout_per_run
    else:
        timeout_per_run = cfg.get("timeout_per_run", 0)
    if timeout_per_run is None or timeout_per_run <= 0:
        timeout_per_run = 0
    datasets = [d.lower().strip() for d in datasets]
    partitions = [p.lower().strip() for p in partitions]
    strategies = [s.lower().strip() for s in strategies]

    raw_name = args.name or cfg.get("name")
    run_name = sanitize_run_name(raw_name) if raw_name else None
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output_dir:
        base_parent = Path(args.output_dir)
        base_output_dir = base_parent / run_name if run_name else base_parent
    elif run_name:
        base_output_dir = Path(cfg.get("output_dir", "results/simulation")) / run_name
    else:
        base_output_dir = Path(cfg.get("output_dir", "results/simulation")) / timestamp_str
    base_output_dir.mkdir(parents=True, exist_ok=True)

    timeout_desc = f"{timeout_per_run}s" if timeout_per_run > 0 else "Disabled (Infinite)"
    print(f"\n{'='*70}")
    print(f"  FedMedAI Automated Simulation Benchmark")
    print(f"{'='*70}")
    print(f"  Run Name         : {run_name or timestamp_str}")
    print(f"  Output Directory : {base_output_dir}")
    print(f"  Model            : {model.upper()}")
    print(f"  Optimizer        : {optimizer.upper()}" + (f" (momentum={momentum})" if optimizer == "sgd" else ""))
    print(f"  Augmentation     : {'Enabled' if augment else 'Disabled'} (crop={random_crop}, flip={random_flip})")
    print(f"  Seed             : {seed}")
    print(f"  Clients          : {num_clients}")
    print(f"  Rounds per Run   : {rounds}")
    print(f"  Local Epochs     : {local_epochs}")
    print(f"  Local Test Eval  : {'Enabled (BMCTA/BTA tracking)' if local_test else 'Disabled (Global Test)'}")
    print(f"  Client Module    : {client_module} ({'Parallel Execution' if client_module == 'client.client' else 'Sequential GPU Lock'})")
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
            p_path = ensure_partition_exists(dataset_name=d, partition_mode=p, num_clients=num_clients, seed=seed, local_test=local_test)
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
                    server_lr_min=server_lr_min,
                    proximal_mu=proximal_mu,
                    early_stop_patience=early_stop_patience,
                    early_stop_metric=early_stop_metric,
                    server_eval=server_eval,
                    local_test=local_test,
                    model=model,
                    seed=seed,
                    optimizer=optimizer,
                    momentum=momentum,
                    weight_decay=weight_decay,
                    augment=augment,
                    random_crop=random_crop,
                    random_flip=random_flip,
                    device_type=device_type,
                    save_client_metrics=save_client_metrics,
                    timeout_per_run=timeout_per_run,
                    run_dir=run_output_dir,
                    config_path=args.config,
                    num_workers=num_workers,
                    client_cpu_threads=client_cpu_threads,
                    client_module=client_module,
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
                        "mcta": rh.get("mcta"),
                        "bmcta": rh.get("bmcta"),
                        "bta": rh.get("bta"),
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
        "run_name": run_name or timestamp_str,
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
            "name": run_name,
            "model": model,
            "seed": seed,
            "rounds": rounds,
            "num_clients": num_clients,
            "local_epochs": local_epochs,
            "lr_mode": lr_mode,
            "local_test": local_test,
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
