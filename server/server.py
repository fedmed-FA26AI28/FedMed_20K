"""FL Server entry point with Step LR Decay, Early Stopping, and YAML config support.

Starts a Flower server with a user-selectable aggregation strategy from
the algorithms/ folder (FedAvg, FedProx, FedNova).

Usage:
    # Run with default settings from configs/experiment.yaml:
    python -m server.server

    # Override parameters via CLI:
    python -m server.server --strategy fedavg --rounds 50 --lr_decay_steps 10 --lr_decay_gamma 0.5

    # FedProx with custom proximal_mu:
    python -m server.server --strategy fedprox --proximal_mu 0.05

    # Early stopping enabled with patience=5:
    python -m server.server --early_stop_patience 5 --early_stop_metric accuracy
"""

import argparse
import json
import os
import sys
import time
import timeit
import signal
import threading
import datetime
from logging import INFO, WARNING
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import flwr as fl
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Scalar,
    Parameters,
)
from flwr.common.logger import log

from models.cnn import CNN, get_parameters, set_parameters
from algorithms import get_strategy
from monitoring.metrics import FLMetricsRecorder
from monitoring.network import get_device_ipv4


# ──────────────────────────────────────────────────────────────
# YAML Config Loader
# ──────────────────────────────────────────────────────────────

def load_yaml_config(config_path: str = "configs/experiment.yaml") -> Dict[str, Any]:
    """Load configuration dictionary from YAML file if available."""
    if not os.path.exists(config_path):
        return {}
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg
    except Exception as e:
        log(WARNING, "Failed to load YAML configuration from %s: %s", config_path, e)
        return {}


# ──────────────────────────────────────────────────────────────
# Interruptible Client Manager (enables Ctrl+C on Windows)
# ──────────────────────────────────────────────────────────────

class InterruptibleClientManager(fl.server.SimpleClientManager):
    """ClientManager subclass that periodically polls with 0.5s timeout.

    On Windows, the default wait_for(timeout=86400) blocks inside a C runtime
    condition variable for 24 hours, preventing Python from processing console
    signals like Ctrl+C / SIGINT. Polling in 0.5s intervals allows Python
    bytecode execution so KeyboardInterrupt is delivered immediately.
    """

    def wait_for(self, num_clients: int, timeout: int = 86400) -> bool:
        """Wait until at least `num_clients` are connected, allowing Ctrl+C."""
        end_time = time.time() + timeout
        while time.time() < end_time:
            with self._cv:
                if len(self.clients) >= num_clients:
                    return True
                self._cv.wait(timeout=0.5)
        with self._cv:
            return len(self.clients) >= num_clients


# ──────────────────────────────────────────────────────────────
# Custom Flower Server with Immediate Early Stopping Support
# ──────────────────────────────────────────────────────────────

class FedMedAIServer(fl.server.Server):
    """Custom Flower Server subclass with active early stopping support and weights preservation."""

    def fit(self, num_rounds: int, timeout: Optional[float] = None) -> Tuple[fl.server.history.History, float]:
        """Run federated averaging with immediate early stopping detection and graceful exit."""
        history = fl.server.history.History()

        # Initialize parameters
        log(INFO, "[INIT]")
        self.parameters = self._get_initial_parameters(server_round=0, timeout=timeout)
        log(INFO, "Starting evaluation of initial global parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])
        else:
            log(INFO, "Evaluation returned no results (`None`)")

        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            # Check if strategy signaled early stopping prior to starting round
            if getattr(self.strategy, "should_stop", False):
                log(
                    WARNING,
                    "[Server] Early stopping condition active before round %d. Stopping.",
                    current_round,
                )
                break

            log(INFO, "")
            log(INFO, "[ROUND %s]", current_round)

            # Train model and replace previous global model
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            # Evaluate model using strategy implementation (centralized server evaluation)
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )
                # Defensively evaluate early stopping if not already tripped
                if hasattr(self.strategy, "check_early_stopping") and not getattr(self.strategy, "should_stop", False):
                    self.strategy.check_early_stopping(
                        server_round=current_round,
                        metrics=metrics_cen,
                        loss=loss_cen,
                        parameters=self.parameters,
                    )

            # Evaluate model on a sample of available clients (federated client evaluation)
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

            # Check if early stopping was triggered during evaluation in this round
            if getattr(self.strategy, "should_stop", False):
                log(
                    WARNING,
                    "[Server] Early stopping triggered at Round %d. Halting training loop.",
                    current_round,
                )
                break

        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        return history, elapsed


# ──────────────────────────────────────────────────────────────
# Server-side evaluate function (centralized evaluation)
# ──────────────────────────────────────────────────────────────

def _get_evaluate_fn(num_classes: int = 8, recorder: Optional[FLMetricsRecorder] = None):
    """Return a server-side evaluation function for centralized evaluation on test set."""
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

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

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
# on_fit_config_fn: send config and LR to clients each round
# ──────────────────────────────────────────────────────────────

def _make_on_fit_config_fn(
    strategy_name: str = "fedavg",
    local_epochs: int = 5,
    learning_rate: float = 0.001,
    lr_mode: str = "client_loss",
    lr_decay_steps: int = 3,
    lr_decay_gamma: float = 0.5,
    proximal_mu: float = 0.0,
    client_lr_patience: int = 2,
    client_lr_factor: float = 0.5,
    client_lr_min: float = 1e-6,
):
    """Create a function that calculates and sends training config to clients each round.

    Supports three LR scheduling modes:
    - 'client_loss': Server sends initial LR on round 1; each client independently reduces LR on local loss plateau.
    - 'server_decay': Server computes global step decay every N rounds and synchronizes across all clients.
    - 'fixed': Constant initial learning rate across all rounds and clients (classic Flower baseline).
    """
    mode = str(lr_mode).lower().strip()

    def on_fit_config(server_round: int) -> Dict[str, Scalar]:
        config: Dict[str, Scalar] = {
            "strategy": strategy_name.lower(),
            "server_round": server_round,
            "local_epochs": local_epochs,
            "lr_mode": mode,
            "initial_lr": float(learning_rate),
        }

        if proximal_mu > 0.0:
            config["proximal_mu"] = proximal_mu

        if mode == "fixed":
            # Constant learning rate for all rounds (classic Flower baseline)
            config["learning_rate"] = float(learning_rate)
            log(
                INFO,
                "[Server] Config for Round %d: strategy=%s, local_epochs=%d, LR=%.6f (fixed)",
                server_round,
                strategy_name.upper(),
                local_epochs,
                learning_rate,
            )
        elif mode in ("server_decay", "server", "step"):
            # Synchronized global step decay: lr = initial_lr * (gamma ** ((round - 1) // steps))
            if lr_decay_steps > 0 and server_round > 1:
                step_count = (server_round - 1) // lr_decay_steps
                scheduled_lr = learning_rate * (lr_decay_gamma ** step_count)
            else:
                scheduled_lr = learning_rate
            config["learning_rate"] = float(scheduled_lr)
            log(
                INFO,
                "[Server] Config for Round %d: strategy=%s, local_epochs=%d, LR=%.6f (server_decay step=%d, gamma=%.2f)",
                server_round,
                strategy_name.upper(),
                local_epochs,
                scheduled_lr,
                lr_decay_steps,
                lr_decay_gamma,
            )
        else:
            # 'client_loss' mode: server sends initial LR on Round 1 only
            config["client_lr_patience"] = int(client_lr_patience)
            config["client_lr_factor"] = float(client_lr_factor)
            config["client_lr_min"] = float(client_lr_min)
            if server_round == 1:
                config["learning_rate"] = float(learning_rate)
                log(
                    INFO,
                    "[Server] Config for Round %d: strategy=%s, local_epochs=%d, init_lr=%.6f (clients reduce LR locally based on loss)",
                    server_round,
                    strategy_name.upper(),
                    local_epochs,
                    learning_rate,
                )
            else:
                log(
                    INFO,
                    "[Server] Config for Round %d: strategy=%s, local_epochs=%d (client_loss mode: clients manage LR on local loss)",
                    server_round,
                    strategy_name.upper(),
                    local_epochs,
                )

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
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment.yaml",
        help="Path to YAML experiment configuration file (default: configs/experiment.yaml)",
    )
    pre_args, remaining_argv = pre_parser.parse_known_args()

    yaml_cfg = load_yaml_config(pre_args.config)

    parser = argparse.ArgumentParser(
        description="FedMedAI FL Server with Configurable LR Scheduling, Early Stopping, and YAML config",
        parents=[pre_parser],
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=yaml_cfg.get("strategy", "fedavg"),
        choices=["fedavg", "fedprox", "fednova", "fedbn", "scaffold"],
        help=f"FL aggregation strategy (default: {yaml_cfg.get('strategy', 'fedavg')})",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=yaml_cfg.get("fl_rounds", 50),
        help=f"Number of FL rounds (default: {yaml_cfg.get('fl_rounds', 50)})",
    )
    parser.add_argument(
        "--min_clients",
        type=int,
        default=yaml_cfg.get("min_clients", 3),
        help=f"Minimum number of clients required to start a round (default: {yaml_cfg.get('min_clients', 3)})",
    )
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=yaml_cfg.get("local_epochs", 5),
        help=f"Number of client local epochs per round (default: {yaml_cfg.get('local_epochs', 5)})",
    )
    parser.add_argument(
        "--lr_mode",
        type=str,
        default=yaml_cfg.get("lr_mode", "client_loss"),
        choices=["client_loss", "server_decay", "fixed"],
        help=f"LR schedule mode: 'client_loss' | 'server_decay' | 'fixed' (default: {yaml_cfg.get('lr_mode', 'client_loss')})",
    )
    parser.add_argument(
        "--learning_rate",
        "--lr",
        type=float,
        default=yaml_cfg.get("learning_rate", 0.001),
        dest="learning_rate",
        help=f"Initial base learning rate (default: {yaml_cfg.get('learning_rate', 0.001)})",
    )
    parser.add_argument(
        "--lr_decay_steps",
        type=int,
        default=yaml_cfg.get("lr_decay_steps", 3),
        help=f"Rounds between LR step decays for 'server_decay' mode (default: {yaml_cfg.get('lr_decay_steps', 3)})",
    )
    parser.add_argument(
        "--lr_decay_gamma",
        type=float,
        default=yaml_cfg.get("lr_decay_gamma", 0.5),
        help=f"Multiplicative factor of LR decay for 'server_decay' mode (default: {yaml_cfg.get('lr_decay_gamma', 0.5)})",
    )
    parser.add_argument(
        "--client_lr_patience",
        type=int,
        default=yaml_cfg.get("client_lr_patience", 2),
        help=f"Client local epochs without loss improvement before reducing LR (default: {yaml_cfg.get('client_lr_patience', 2)})",
    )
    parser.add_argument(
        "--client_lr_factor",
        type=float,
        default=yaml_cfg.get("client_lr_factor", 0.5),
        help=f"Multiplicative factor for client LR reduction (default: {yaml_cfg.get('client_lr_factor', 0.5)})",
    )
    parser.add_argument(
        "--client_lr_min",
        type=float,
        default=yaml_cfg.get("client_lr_min", 1e-6),
        help=f"Minimum client learning rate bound (default: {yaml_cfg.get('client_lr_min', 1e-6)})",
    )
    parser.add_argument(
        "--proximal_mu",
        type=float,
        default=yaml_cfg.get("proximal_mu", 0.01),
        help=f"FedProx proximal mu parameter (default: {yaml_cfg.get('proximal_mu', 0.01)})",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=yaml_cfg.get("early_stop_patience", 10),
        help=f"Server-side early stopping patience in rounds, 0 to disable (default: {yaml_cfg.get('early_stop_patience', 10)})",
    )
    parser.add_argument(
        "--early_stop_metric",
        type=str,
        default=yaml_cfg.get("early_stop_metric", "accuracy"),
        choices=["accuracy", "loss"],
        help=f"Target metric for early stopping (default: {yaml_cfg.get('early_stop_metric', 'accuracy')})",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=yaml_cfg.get("server_host", "0.0.0.0"),
        help=f"Server listen address (default: {yaml_cfg.get('server_host', '0.0.0.0')})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=yaml_cfg.get("server_port", 8080),
        help=f"Server listen port (default: {yaml_cfg.get('server_port', 8080)})",
    )
    parser.add_argument(
        "--server_eval",
        dest="server_eval",
        action=argparse.BooleanOptionalAction,
        default=yaml_cfg.get("server_eval", True),
        help=f"Enable server-side centralized evaluation (default: {yaml_cfg.get('server_eval', True)})",
    )
    args = parser.parse_args()

    server_ipv4 = get_device_ipv4()

    print(f"\n{'='*60}")
    print(f"  FedMedAI FL Server")
    print(f"  Config Source: {pre_args.config}")
    print(f"  Strategy     : {args.strategy.upper()}")
    print(f"  Server IPv4  : {server_ipv4}")
    print(f"  Client Target: {server_ipv4}:{args.port}")
    print(f"  Rounds       : {args.rounds}")
    print(f"  Min Clients  : {args.min_clients}")
    print(f"  Local Epochs : {args.local_epochs}")
    print(f"  LR Mode      : {args.lr_mode.upper()}")
    print(f"  Base LR      : {args.learning_rate}")
    if args.lr_mode in ("server_decay", "server", "step"):
        print(f"  Server Decay : Step decay every {args.lr_decay_steps} rounds (gamma={args.lr_decay_gamma})")
    elif args.lr_mode == "client_loss":
        print(f"  Client Reducer: Initial LR passed on Round 1 | patience={args.client_lr_patience}, factor={args.client_lr_factor}, min={args.client_lr_min}")
    else:
        print(f"  LR Policy    : Fixed constant learning rate across all rounds")
    if args.strategy == "fedprox":
        print(f"  Proximal mu  : {args.proximal_mu}")
    print(f"  Early Stop   : Patience={args.early_stop_patience} (metric: {args.early_stop_metric})")
    print(f"  Server Eval  : {args.server_eval}")
    print(f"  Listen       : {args.host}:{args.port}")
    print(f"{'='*60}\n")

    initial_model = CNN(num_classes=8)
    initial_params = ndarrays_to_parameters(get_parameters(initial_model))

    recorder = FLMetricsRecorder(
        strategy_name=args.strategy,
        proximal_mu=args.proximal_mu if args.strategy == "fedprox" else None,
    )

    strategy_kwargs = {
        "min_fit_clients": args.min_clients,
        "min_evaluate_clients": args.min_clients,
        "min_available_clients": args.min_clients,
        "initial_parameters": initial_params,
        "metrics_recorder": recorder,
        "on_fit_config_fn": _make_on_fit_config_fn(
            strategy_name=args.strategy,
            local_epochs=args.local_epochs,
            learning_rate=args.learning_rate,
            lr_mode=args.lr_mode,
            lr_decay_steps=args.lr_decay_steps,
            lr_decay_gamma=args.lr_decay_gamma,
            proximal_mu=args.proximal_mu if args.strategy == "fedprox" else 0.0,
            client_lr_patience=args.client_lr_patience,
            client_lr_factor=args.client_lr_factor,
            client_lr_min=args.client_lr_min,
        ),
        "evaluate_metrics_aggregation_fn": _evaluate_metrics_aggregation_fn,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_metric": args.early_stop_metric,
    }

    if args.server_eval:
        strategy_kwargs["evaluate_fn"] = _get_evaluate_fn(num_classes=8, recorder=recorder)

    if args.strategy == "fedprox":
        strategy_kwargs["proximal_mu"] = args.proximal_mu
    elif args.strategy == "fedbn":
        strategy_kwargs["model"] = initial_model
    elif args.strategy == "scaffold":
        strategy_kwargs["num_total_clients"] = args.min_clients
        strategy_kwargs["model"] = initial_model

    strategy = get_strategy(args.strategy, **strategy_kwargs)

    print(f"[Server] Strategy created: {type(strategy).__name__}")
    print(f"[Server] Waiting for {args.min_clients} clients on {args.host}:{args.port} (Connect target: {server_ipv4}:{args.port})...\n")

    # Instantiate custom server with early stopping halt capability and interruptible client manager
    server = FedMedAIServer(
        client_manager=InterruptibleClientManager(),
        strategy=strategy,
    )

    total_start = time.time()

    try:
        history = fl.server.start_server(
            server_address=f"{args.host}:{args.port}",
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            server=server,
        )
    except KeyboardInterrupt:
        print("\n[Server] Interrupted by user (Ctrl+C). Shutting down FL server gracefully...")
        sys.exit(0)

    total_time = time.time() - total_start

    summary = strategy.get_summary()
    print(f"\n{'='*60}")
    print(f"  FL Training Complete!")
    print(f"{'='*60}")
    print(f"  Strategy       : {args.strategy.upper()}")
    print(f"  Total Rounds   : {summary.get('total_rounds', args.rounds)}")
    print(f"  Total Time     : {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Avg Round Time : {summary.get('avg_round_time_seconds', 0):.1f}s")
    print(f"  Best Metric    : {summary.get('best_metric', 'N/A')} (Round {summary.get('best_round', 'N/A')})")
    print(f"  Early Stopped  : {summary.get('early_stopped', False)}")
    if args.strategy == "fedprox":
        print(f"  Proximal mu    : {summary.get('proximal_mu', args.proximal_mu)}")
    elif args.strategy == "fedbn":
        print(f"  Local BN Layers: {summary.get('num_bn_layers_local', 'N/A')}")
        print(f"  Aggregated Non-BN: {summary.get('num_non_bn_layers_aggregated', 'N/A')}")
    elif args.strategy == "scaffold":
        print(f"  Final Server c Norm: {summary.get('server_c_norm', 'N/A')}")
    print(f"{'='*60}\n")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results/federated/{args.strategy}/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    extra_metadata = {
        "strategy": args.strategy,
        "num_rounds_requested": args.rounds,
        "min_clients": args.min_clients,
        "local_epochs": args.local_epochs,
        "initial_learning_rate": args.learning_rate,
        "lr_decay_steps": args.lr_decay_steps,
        "lr_decay_gamma": args.lr_decay_gamma,
        "server_eval_enabled": args.server_eval,
        "server_address": f"{args.host}:{args.port}",
        "early_stop_patience": args.early_stop_patience,
        "early_stop_metric": args.early_stop_metric,
        "early_stopped": summary.get("early_stopped", False),
        "best_metric": summary.get("best_metric"),
        "best_round": summary.get("best_round"),
    }
    if args.strategy == "fedprox":
        extra_metadata["proximal_mu"] = args.proximal_mu
    elif args.strategy == "fedbn":
        extra_metadata["num_bn_layers_local"] = summary.get("num_bn_layers_local")
        extra_metadata["num_non_bn_layers_aggregated"] = summary.get("num_non_bn_layers_aggregated")
    elif args.strategy == "scaffold":
        extra_metadata["server_c_norm"] = summary.get("server_c_norm")

    print("[Server] Generating and saving CSV metrics, comparison plots, and rich fl_results.json...")
    saved_artifacts = strategy.save_artifacts(save_dir=save_dir, extra_metadata=extra_metadata)

    # Save model weights checkpoints (best weights and final round weights)
    best_params = strategy.get_best_parameters()
    if best_params is not None:
        best_model = CNN(num_classes=8)
        set_parameters(best_model, parameters_to_ndarrays(best_params))
        best_model_path = os.path.join(save_dir, "best_model.pth")
        torch.save(best_model.state_dict(), best_model_path)
        print(f"  Saved Best Model Checkpoint: {os.path.basename(best_model_path)}")

    if server.parameters is not None:
        final_model = CNN(num_classes=8)
        set_parameters(final_model, parameters_to_ndarrays(server.parameters))
        final_model_path = os.path.join(save_dir, "final_model.pth")
        torch.save(final_model.state_dict(), final_model_path)
        print(f"  Saved Final Model Checkpoint: {os.path.basename(final_model_path)}")

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
