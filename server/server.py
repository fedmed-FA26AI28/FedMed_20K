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
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Scalar

from models.cnn import CNN, build_model, get_parameters, set_parameters
from algorithms import get_strategy


from monitoring.metrics import FLMetricsRecorder


# ──────────────────────────────────────────────────────────────
# Server-side evaluate function (centralized evaluation)
# ──────────────────────────────────────────────────────────────

def _get_evaluate_fn(num_classes: int = 8, recorder: Optional[FLMetricsRecorder] = None,
                     model_name: str = "legacy", size: int = 28):
    """Return an optional per-round centralized validation function.

    The test split is deliberately not loaded here. It is reserved for the
    single final evaluation after federated training has completed.
    """
    import torch
    from datasets.medmnist_code import get_bloodmnist_dataset
    from torch.utils.data import DataLoader

    options = {"size": size} if size != 28 else {}
    val_dataset, _ = get_bloodmnist_dataset("val", download=True, **options)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(
        server_round: int,
        parameters_ndarrays: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        model = build_model(model_name, num_classes=num_classes).to(device)
        from models.cnn import set_parameters
        set_parameters(model, parameters_ndarrays)
        model.eval()

        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
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
            f"  [Server] Round {server_round} Global Validation\n"
            f"  Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n"
            f"{'='*60}\n"
        )

        return float(avg_loss), {"accuracy": float(accuracy)}

    return evaluate


def _evaluate_final_global_model(
    parameters_ndarrays: List[np.ndarray],
    num_classes: int = 8,
    calibration_indices: Optional[List[int]] = None,
    conformal_alpha: float = 0.1,
    model_name: str = "legacy",
    size: int = 28,
    normalization=None,
) -> Dict[str, object]:
    """Calibrate on held-out validation, then evaluate test exactly once."""
    import torch
    from datasets.medmnist_code import get_bloodmnist_dataset
    from models.cnn import build_model, set_parameters
    from monitoring.reliability import (
        collect_logits,
        conformal_metrics,
        fit_conformal_threshold,
        fit_temperature,
        probability_metrics,
    )
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    from torch.utils.data import DataLoader, Subset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, num_classes=num_classes).to(device)
    set_parameters(model, parameters_ndarrays)
    dataset_options = {}
    if size != 28:
        dataset_options["size"] = size
    if normalization is not None:
        dataset_options["normalization"] = normalization

    temperature = 1.0
    conformal_threshold = None
    calibration_size = 0
    if calibration_indices is not None:
        validation_dataset, _ = get_bloodmnist_dataset("val", download=True, **dataset_options)
        calibration_loader = DataLoader(
            Subset(validation_dataset, calibration_indices),
            batch_size=64,
            shuffle=False,
            num_workers=0,
        )
        calibration_logits, calibration_labels = collect_logits(
            model, calibration_loader, device
        )
        calibration_size = len(calibration_labels)
        temperature = fit_temperature(calibration_logits, calibration_labels)
        conformal_threshold = fit_conformal_threshold(
            calibration_logits,
            calibration_labels,
            alpha=conformal_alpha,
            temperature=temperature,
        )

    # The test split is not loaded until FL and validation/calibration finish.
    test_dataset, _ = get_bloodmnist_dataset("test", download=True, **dataset_options)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_logits, test_labels = collect_logits(model, test_loader, device)
    predictions = test_logits.argmax(dim=1)
    y_true = test_labels.tolist()
    y_pred = predictions.tolist()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), average="macro", zero_division=0
    )
    _, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0
    )
    accuracy = predictions.eq(test_labels).float().mean().item()
    result: Dict[str, object] = {
        "loss": float(torch.nn.functional.cross_entropy(test_logits, test_labels).item()),
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "balanced_accuracy": float(recall),
        "worst_class_recall": float(min(per_class_recall)),
        "per_class_recall": [float(value) for value in per_class_recall],
        "per_class_f1": [float(value) for value in per_class_f1],
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(num_classes))
        ).tolist(),
        "num_samples": int(len(test_labels)),
        "uncalibrated": probability_metrics(test_logits, test_labels),
    }
    if conformal_threshold is not None:
        result.update(
            {
                "calibration_num_samples": calibration_size,
                "temperature": temperature,
                "calibrated": probability_metrics(
                    test_logits, test_labels, temperature=temperature
                ),
                "conformal_alpha": conformal_alpha,
                "conformal_threshold": conformal_threshold,
                "conformal": conformal_metrics(
                    test_logits,
                    test_labels,
                    threshold=conformal_threshold,
                    temperature=temperature,
                ),
            }
        )
    return result


# ──────────────────────────────────────────────────────────────
# on_fit_config_fn: send config to clients each round
# ──────────────────────────────────────────────────────────────

def _make_on_fit_config_fn(
    local_epochs: int,
    learning_rate: float,
    proximal_mu: float = 0.0,
    logit_tau: float = 0.0,
    prior_smoothing: float = 1.0,
    head_mu: float = 0.0,
    coverage_kappa: float = 32.0,
    balanced_sampling: bool = False,
):
    """Create a function that sends training config to clients each round."""

    def on_fit_config(server_round: int) -> Dict[str, Scalar]:
        config = {
            "server_round": server_round,
            "local_epochs": local_epochs,
            "learning_rate": learning_rate,
            "balanced_sampling": balanced_sampling,
        }
        if proximal_mu > 0.0:
            config["proximal_mu"] = proximal_mu
        if logit_tau > 0.0 or head_mu > 0.0:
            config.update(
                {
                    "logit_tau": logit_tau,
                    "prior_smoothing": prior_smoothing,
                    "head_mu": head_mu,
                    "coverage_kappa": coverage_kappa,
                }
            )
        return config

    return on_fit_config


def _make_on_evaluate_config_fn():
    """Send round identity to clients for validation logging only."""

    def on_evaluate_config(server_round: int) -> Dict[str, Scalar]:
        return {"server_round": server_round}

    return on_evaluate_config


# ──────────────────────────────────────────────────────────────
# Evaluate metrics aggregation (federated evaluation)
# ──────────────────────────────────────────────────────────────

def _evaluate_metrics_aggregation_fn(
    eval_metrics: List[Tuple[int, Dict[str, Scalar]]],
) -> Dict[str, Scalar]:
    """Aggregate mean and worst-client validation performance."""
    total_examples = sum(n for n, _ in eval_metrics)
    result: Dict[str, Scalar] = {}
    for key in ("accuracy", "f1_macro", "balanced_accuracy", "worst_class_recall"):
        weighted = sum(
            n * float(metrics.get(key, 0.0)) for n, metrics in eval_metrics
        )
        result[key] = weighted / total_examples if total_examples > 0 else 0.0
    if eval_metrics:
        result["worst_client_accuracy"] = min(
            float(metrics.get("accuracy", 0.0)) for _, metrics in eval_metrics
        )
        result["worst_client_f1_macro"] = min(
            float(metrics.get("f1_macro", 0.0)) for _, metrics in eval_metrics
        )
    return result


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FedMedAI FL Server")
    parser.add_argument(
        "--strategy",
        type=str,
        default="fedavg",
        choices=["fedavg", "fedprox", "fednova", "coverage", "balanced", "logit_only", "head_only"],
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
    parser.add_argument("--logit_tau", type=float, default=1.0)
    parser.add_argument("--prior_smoothing", type=float, default=1.0)
    parser.add_argument("--head_mu", type=float, default=0.01)
    parser.add_argument("--coverage_kappa", type=float, default=32.0)
    parser.add_argument("--model", choices=["legacy", "tiny_cnn", "mobilenet_v3_small"], default="legacy")
    parser.add_argument("--size", type=int, choices=[28, 64], default=28)
    parser.add_argument("--final_test", action="store_true",
                        help="Evaluate held-out test after a locked final run")
    parser.add_argument("--locked_config", default=None,
                        help="Development run_spec.json that fixes strategy/model/size")
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Server-side early stopping patience in rounds (0=disable, default: 0)",
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
        help="Enable server-side centralized validation each round",
    )
    args = parser.parse_args()
    if args.final_test:
        if not args.locked_config:
            parser.error("--final_test requires --locked_config")
        with open(args.locked_config, encoding="utf-8") as file:
            locked = json.load(file)
        locked_fields = {
            "strategy": args.strategy, "model": args.model, "size": args.size,
            "rounds": args.rounds, "local_epochs": args.local_epochs,
            "num_clients": args.min_clients,
            "learning_rate": args.learning_rate,
            "proximal_mu": args.proximal_mu,
            "logit_tau": args.logit_tau, "head_mu": args.head_mu,
            "coverage_kappa": args.coverage_kappa,
            "early_stop_patience": args.early_stop_patience,
        }
        for key, value in locked_fields.items():
            if locked.get(key) != value:
                parser.error(f"{key} differs from locked development run")

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
    initial_model = build_model(args.model, num_classes=8)
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
            logit_tau=args.logit_tau if args.strategy in {"coverage", "logit_only"} else 0.0,
            prior_smoothing=args.prior_smoothing,
            head_mu=args.head_mu if args.strategy in {"coverage", "head_only"} else 0.0,
            coverage_kappa=args.coverage_kappa,
            balanced_sampling=args.strategy == "balanced",
        ),
        "evaluate_metrics_aggregation_fn": _evaluate_metrics_aggregation_fn,
        "on_evaluate_config_fn": _make_on_evaluate_config_fn(),
        "early_stop_patience": args.early_stop_patience,
    }

    # Server-side evaluation (optional)
    if args.server_eval:
        strategy_kwargs["evaluate_fn"] = _get_evaluate_fn(
            num_classes=8, recorder=recorder, model_name=args.model, size=args.size
        )

    # Strategy-specific kwargs
    if args.strategy == "fedprox":
        strategy_kwargs["proximal_mu"] = args.proximal_mu

    # Create strategy from algorithms/ folder
    strategy = get_strategy(args.strategy, **strategy_kwargs)
    strategy.latest_parameters = initial_params

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

    final_parameters = parameters_to_ndarrays(strategy.latest_parameters)
    final_test_metrics = None
    if args.final_test:
        final_test_metrics = _evaluate_final_global_model(
            final_parameters, num_classes=8, model_name=args.model, size=args.size
        )

    # Print summary
    summary = strategy.get_summary()
    print(f"\n{'='*60}")
    print(f"  FL Training Complete!")
    print(f"{'='*60}")
    print(f"  Strategy       : {args.strategy.upper()}")
    print(f"  Total Rounds   : {summary.get('total_rounds', args.rounds)}")
    print(f"  Total Time     : {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Avg Round Time : {summary.get('avg_round_time_seconds', 0):.1f}s")
    print(f"  Best Val Metric: {summary.get('best_metric', 'N/A')}")
    if final_test_metrics is not None:
        print(f"  Final Test Loss: {final_test_metrics['loss']:.4f}")
        print(f"  Final Test Acc : {final_test_metrics['accuracy']:.4f}")
        print(f"  Final Test F1  : {final_test_metrics['f1_macro']:.4f} (macro)")
    if args.strategy == "fedprox":
        print(f"  Proximal mu    : {summary.get('proximal_mu', args.proximal_mu)}")
    print(f"{'='*60}\n")

    # Save comprehensive results, CSVs, plots, and JSON
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results/federated/{args.strategy}/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    extra_metadata = {
        "strategy": args.strategy,
        "model": args.model,
        "size": args.size,
        "final_test_enabled": args.final_test,
        "num_rounds_requested": args.rounds,
        "min_clients": args.min_clients,
        "local_epochs": args.local_epochs,
        "learning_rate": args.learning_rate,
        "server_eval_enabled": args.server_eval,
        "server_validation_enabled": args.server_eval,
        "server_address": f"{args.host}:{args.port}",
        "early_stop_patience": args.early_stop_patience,
        "final_test_metrics": final_test_metrics,
    }
    if args.strategy == "fedprox":
        extra_metadata["proximal_mu"] = args.proximal_mu

    print("[Server] Generating and saving all CSV metrics, comparison plots, and rich fl_results.json...")
    saved_artifacts = strategy.save_artifacts(save_dir=save_dir, extra_metadata=extra_metadata)
    global_model = build_model(args.model, num_classes=8)
    set_parameters(global_model, final_parameters)
    import torch
    torch.save(global_model.state_dict(), os.path.join(save_dir, "global_model.pt"))
    final_test_path = None
    if final_test_metrics is not None:
        final_test_path = os.path.join(save_dir, "final_test_metrics.json")
        with open(final_test_path, "w", encoding="utf-8") as f:
            json.dump(final_test_metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  FedMedAI FL Artifacts Saved Successfully!")
    print(f"{'='*60}")
    print(f"  Output Directory : {save_dir}")
    print(f"  Summary JSON     : {saved_artifacts.get('json_path')}")
    print(f"  Final Test JSON  : {final_test_path}")
    print(f"  CSV Metrics      :")
    for csv_key, csv_file in saved_artifacts.get("csv_paths", {}).items():
        print(f"    - {csv_key}: {os.path.basename(csv_file)}")
    print(f"  Comparison Plots :")
    for plot_file in saved_artifacts.get("plot_paths", []):
        print(f"    - {os.path.basename(plot_file)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
