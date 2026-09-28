"""FL Server entry point.

Starts a Flower server with a user-selectable aggregation strategy from
the algorithms/ folder (FedAvg, FedProx, FedNova).

Usage:
    # FedAvg (default)
    python -m server.server --strategy fedavg --rounds 50 --min_clients 3

    # FedProx with mu=0.1
    python -m server.server --strategy fedprox --proximal_mu 0.1 --rounds 50

    # FedNova
    python -m server.server --strategy fednova --rounds 50 --min_clients 10

    # Custom host/port for LAN deployment
    python -m server.server --host 0.0.0.0 --port 8080 --min_clients 10
"""

import argparse
import json
import os
import time
import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import flwr as fl
from flwr.common import ndarrays_to_parameters, Scalar

from models.cnn import CNN, get_parameters
from algorithms import get_strategy


from monitoring.metrics import FLMetricsRecorder


# ──────────────────────────────────────────────────────────────
# Server-side evaluate function (centralized evaluation)
# ──────────────────────────────────────────────────────────────

def _get_evaluate_fn(num_classes: int = 8, recorder: Optional[FLMetricsRecorder] = None):
    """Return a server-side evaluation function (optional).

    If you want the server to also evaluate the global model on a held-out
    test set after each round, provide this. Otherwise set evaluate_fn=None
    and rely on federated (client-side) evaluation only.
    """
    import torch
    from datasets.medmnist_code import get_bloodmnist_datasets
    from torch.utils.data import DataLoader

    _, _, test_dataset, _ = get_bloodmnist_datasets(download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(
        server_round: int,
        parameters_ndarrays: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        model = CNN(num_classes=num_classes).to(device)
        from models.cnn import set_parameters
        set_parameters(model, parameters_ndarrays)
        model.eval()

        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.squeeze().long().to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total

        if recorder is not None:
            recorder.record_eval_results(
                server_round=server_round,
                loss=avg_loss,
                accuracy=accuracy,
                is_server_eval=True,
            )

        print(
            f"\n{'='*60}\n"
            f"  [Server] Round {server_round} Global Evaluation\n"
            f"  Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n"
            f"{'='*60}\n"
        )

        return float(avg_loss), {"accuracy": float(accuracy)}

    return evaluate


# ──────────────────────────────────────────────────────────────
# on_fit_config_fn: send config to clients each round
# ──────────────────────────────────────────────────────────────

def _make_on_fit_config_fn(
    local_epochs: int,
    learning_rate: float,
    proximal_mu: float = 0.0,
):
    """Create a function that sends training config to clients each round."""

    def on_fit_config(server_round: int) -> Dict[str, Scalar]:
        config = {
            "server_round": server_round,
            "local_epochs": local_epochs,
            "learning_rate": learning_rate,
        }
        if proximal_mu > 0.0:
            config["proximal_mu"] = proximal_mu
        return config

    return on_fit_config


# ──────────────────────────────────────────────────────────────
# Evaluate metrics aggregation (federated evaluation)
# ──────────────────────────────────────────────────────────────

def _evaluate_metrics_aggregation_fn(
    eval_metrics: List[Tuple[int, Dict[str, Scalar]]],
) -> Dict[str, Scalar]:
    """Aggregate evaluation metrics from clients (weighted average accuracy)."""
    total_examples = sum(n for n, _ in eval_metrics)
    weighted_acc = sum(
        n * float(m.get("accuracy", 0.0)) for n, m in eval_metrics
    )
    return {"accuracy": weighted_acc / total_examples if total_examples > 0 else 0.0}


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FedMedAI FL Server")
    parser.add_argument(
        "--strategy",
        type=str,
        default="fedavg",
        choices=["fedavg", "fedprox", "fednova"],
        help="FL aggregation strategy (default: fedavg)",
    )
    parser.add_argument(
        "--rounds", type=int, default=50, help="Number of FL rounds (default: 50)"
    )
    parser.add_argument(
        "--min_clients",
        type=int,
        default=3,
        help="Minimum number of clients before starting a round (default: 3)",
    )
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=5,
        help="Local training epochs per round (default: 5)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate sent to clients (default: 0.001)",
    )
    parser.add_argument(
        "--proximal_mu",
        type=float,
        default=0.1,
        help="FedProx proximal mu (only used when --strategy=fedprox, default: 0.1)",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=10,
        help="Server-side early stopping patience in rounds (0=disable, default: 10)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server listen address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Server listen port (default: 8080)",
    )
    parser.add_argument(
        "--server_eval",
        action="store_true",
        help="Enable server-side centralized evaluation each round",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  FedMedAI FL Server")
    print(f"  Strategy : {args.strategy.upper()}")
    print(f"  Rounds   : {args.rounds}")
    print(f"  Min Clients: {args.min_clients}")
    print(f"  Local Epochs: {args.local_epochs}")
    print(f"  LR       : {args.learning_rate}")
    if args.strategy == "fedprox":
        print(f"  Proximal mu: {args.proximal_mu}")
    print(f"  Early Stop Patience: {args.early_stop_patience}")
    print(f"  Listen   : {args.host}:{args.port}")
    print(f"{'='*60}\n")

    # Create initial model parameters
    initial_model = CNN(num_classes=8)
    initial_params = ndarrays_to_parameters(get_parameters(initial_model))

    # Initialize metrics recorder
    recorder = FLMetricsRecorder(
        strategy_name=args.strategy,
        proximal_mu=args.proximal_mu if args.strategy == "fedprox" else None,
    )

    # Build strategy kwargs
    strategy_kwargs = {
        "min_fit_clients": args.min_clients,
        "min_evaluate_clients": args.min_clients,
        "min_available_clients": args.min_clients,
        "initial_parameters": initial_params,
        "metrics_recorder": recorder,
        "on_fit_config_fn": _make_on_fit_config_fn(
            local_epochs=args.local_epochs,
            learning_rate=args.learning_rate,
            proximal_mu=args.proximal_mu if args.strategy == "fedprox" else 0.0,
        ),
        "evaluate_metrics_aggregation_fn": _evaluate_metrics_aggregation_fn,
        "early_stop_patience": args.early_stop_patience,
    }

    # Server-side evaluation (optional)
    if args.server_eval:
        strategy_kwargs["evaluate_fn"] = _get_evaluate_fn(num_classes=8, recorder=recorder)

    # Strategy-specific kwargs
    if args.strategy == "fedprox":
        strategy_kwargs["proximal_mu"] = args.proximal_mu

    # Create strategy from algorithms/ folder
    strategy = get_strategy(args.strategy, **strategy_kwargs)

    print(f"[Server] Strategy created: {type(strategy).__name__}")
    print(f"[Server] Waiting for {args.min_clients} clients on {args.host}:{args.port}...\n")

    # Start FL server
    total_start = time.time()

    history = fl.server.start_server(
        server_address=f"{args.host}:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    total_time = time.time() - total_start

    # Print summary
    summary = strategy.get_summary()
    print(f"\n{'='*60}")
    print(f"  FL Training Complete!")
    print(f"{'='*60}")
    print(f"  Strategy       : {args.strategy.upper()}")
    print(f"  Total Rounds   : {summary.get('total_rounds', args.rounds)}")
    print(f"  Total Time     : {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Avg Round Time : {summary.get('avg_round_time_seconds', 0):.1f}s")
    print(f"  Best Metric    : {summary.get('best_metric', 'N/A')}")
    if args.strategy == "fedprox":
        print(f"  Proximal mu    : {summary.get('proximal_mu', args.proximal_mu)}")
    print(f"{'='*60}\n")

    # Save comprehensive results, CSVs, plots, and JSON
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results/federated/{args.strategy}/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    extra_metadata = {
        "strategy": args.strategy,
        "num_rounds_requested": args.rounds,
        "min_clients": args.min_clients,
        "local_epochs": args.local_epochs,
        "learning_rate": args.learning_rate,
        "server_eval_enabled": args.server_eval,
        "server_address": f"{args.host}:{args.port}",
        "early_stop_patience": args.early_stop_patience,
    }
    if args.strategy == "fedprox":
        extra_metadata["proximal_mu"] = args.proximal_mu

    print("[Server] Generating and saving all CSV metrics, comparison plots, and rich fl_results.json...")
    saved_artifacts = strategy.save_artifacts(save_dir=save_dir, extra_metadata=extra_metadata)

    print(f"\n{'='*60}")
    print(f"  FedMedAI FL Artifacts Saved Successfully!")
    print(f"{'='*60}")
    print(f"  Output Directory : {save_dir}")
    print(f"  Summary JSON     : {saved_artifacts.get('json_path')}")
    print(f"  CSV Metrics      :")
    for csv_key, csv_file in saved_artifacts.get("csv_paths", {}).items():
        print(f"    - {csv_key}: {os.path.basename(csv_file)}")
    print(f"  Comparison Plots :")
    for plot_file in saved_artifacts.get("plot_paths", []):
        print(f"    - {os.path.basename(plot_file)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
