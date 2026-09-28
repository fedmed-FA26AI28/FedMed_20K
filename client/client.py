"""FL Client entry point.

Implements a Flower NumPyClient that:
- Loads a local data partition (from a pre-generated Dirichlet JSON).
- Trains the CNN model locally with LR scheduling and early stopping.
- Reports extended metrics: per-epoch time, total training time, weight size.
- Supports FedProx proximal term when instructed by the server config.

Usage:
    python -m client.client --client_id 0 --server_address 127.0.0.1:8080
    python -m client.client --client_id 5 --server_address 192.168.1.15:8080 --partition_path data/partitions/partition_seed42_alpha0.3_clients10.json
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import flwr as fl
from flwr.common import NDArrays, Scalar

from models.cnn import CNN, get_parameters, set_parameters
from datasets.medmnist_code import get_bloodmnist_datasets
from datasets.partition import load_partition, get_client_dataloader


# ──────────────────────────────────────────────────────────────
# Local training function (FL-aware version with proximal term)
# ──────────────────────────────────────────────────────────────

def _local_train(
    model: nn.Module,
    train_loader,
    optimizer,
    device: torch.device,
    epochs: int,
    proximal_mu: float = 0.0,
    global_params: list = None,
    min_lr: float = 1e-6,
    early_stop_patience: int = 5,
):
    """Train locally and return rich metrics.

    Args:
        proximal_mu: If > 0 and global_params provided, adds FedProx proximal term.
        global_params: Global model parameters (list of tensors) for proximal term.
    """
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=min_lr
    )

    best_loss = float("inf")
    patience_counter = 0
    epoch_times = []
    total_samples = 0

    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # FedProx proximal term
            if proximal_mu > 0.0 and global_params is not None:
                proximal_loss = 0.0
                for local_p, global_p in zip(
                    model.parameters(), global_params
                ):
                    proximal_loss += ((local_p - global_p) ** 2).sum()
                loss = loss + (proximal_mu / 2.0) * proximal_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        total_samples = total

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  [Client] Epoch [{epoch+1}/{epochs}] "
            f"LR={current_lr:.6f} | Loss={epoch_loss:.4f} Acc={epoch_acc:.4f} "
            f"| {epoch_time:.1f}s"
        )

        # LR scheduling on train loss
        old_lr = current_lr
        scheduler.step(epoch_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < old_lr:
            print(f"    -> LR reduced: {old_lr:.6f} -> {new_lr:.6f}")

        # Early stopping on train loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(
                    f"    -> Early stopping at epoch {epoch+1} "
                    f"(no improvement for {early_stop_patience} epochs)"
                )
                break

    return {
        "train_loss": epoch_loss,
        "train_accuracy": epoch_acc,
        "epoch_times": epoch_times,
        "epoch_time_avg": float(np.mean(epoch_times)),
        "num_samples": total_samples,
        "epochs_run": len(epoch_times),
    }


# ──────────────────────────────────────────────────────────────
# Flower NumPyClient
# ──────────────────────────────────────────────────────────────

class FedMedAIClient(fl.client.NumPyClient):
    """Flower client for FedMedAI federated learning."""

    def __init__(
        self,
        client_id: int,
        train_loader,
        test_loader,
        num_classes: int = 8,
        local_epochs: int = 5,
        learning_rate: float = 0.001,
    ):
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = CNN(num_classes=num_classes).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        print(
            f"[Client {self.client_id}] Initialized on {self.device} | "
            f"train_samples={len(train_loader.dataset)}"
        )

    def get_parameters(self, config) -> NDArrays:
        """Return model parameters as numpy arrays."""
        return get_parameters(self.model)

    def fit(self, parameters: NDArrays, config: dict):
        """Receive global model, train locally, return updated weights + metrics."""
        # Load global parameters
        set_parameters(self.model, parameters)

        # Read config from server
        local_epochs = int(config.get("local_epochs", self.local_epochs))
        proximal_mu = float(config.get("proximal_mu", 0.0))
        lr_override = config.get("learning_rate")
        if lr_override is not None:
            for pg in self.optimizer.param_groups:
                pg["lr"] = float(lr_override)

        # Prepare global params for FedProx proximal term
        global_params = None
        if proximal_mu > 0.0:
            global_params = [
                p.clone().detach() for p in self.model.parameters()
            ]

        # Train
        train_start = time.time()
        train_metrics = _local_train(
            model=self.model,
            train_loader=self.train_loader,
            optimizer=self.optimizer,
            device=self.device,
            epochs=local_epochs,
            proximal_mu=proximal_mu,
            global_params=global_params,
        )
        training_time = time.time() - train_start

        # Compute weight size in KB
        updated_params = get_parameters(self.model)
        weight_size_bytes = sum(p.nbytes for p in updated_params)
        weight_size_kb = weight_size_bytes / 1024.0

        # Build metrics dict (only Scalar types: bool, bytes, float, int, str)
        metrics = {
            "client_id": self.client_id,
            "train_loss": float(train_metrics["train_loss"]),
            "train_accuracy": float(train_metrics["train_accuracy"]),
            "training_time": float(training_time),
            "epoch_time_avg": float(train_metrics["epoch_time_avg"]),
            "epochs_run": train_metrics["epochs_run"],
            "local_steps": train_metrics["epochs_run"],
            "weight_size_kb": float(weight_size_kb),
        }

        print(
            f"[Client {self.client_id}] fit done | "
            f"loss={metrics['train_loss']:.4f} acc={metrics['train_accuracy']:.4f} "
            f"| {training_time:.1f}s | weights={weight_size_kb:.1f}KB"
        )

        return updated_params, train_metrics["num_samples"], metrics

    def evaluate(self, parameters: NDArrays, config: dict):
        """Evaluate the global model on the local test set."""
        set_parameters(self.model, parameters)
        self.model.eval()

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.squeeze().long().to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total

        print(
            f"[Client {self.client_id}] evaluate | "
            f"loss={avg_loss:.4f} acc={accuracy:.4f}"
        )

        return float(avg_loss), total, {"accuracy": float(accuracy)}


# ──────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FedMedAI FL Client")
    parser.add_argument(
        "--client_id", type=int, required=True, help="Client ID (0-9)"
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default="127.0.0.1:8080",
        help="FL Server address (default: 127.0.0.1:8080)",
    )
    parser.add_argument(
        "--partition_path",
        type=str,
        default=None,
        help="Path to partition JSON. If None, auto-detect from data/partitions/",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.3, help="Dirichlet alpha (default: 0.3)"
    )
    parser.add_argument(
        "--num_clients", type=int, default=3, help="Total number of clients (default: 3)"
    )
    parser.add_argument(
        "--local_epochs", type=int, default=5, help="Local training epochs per round"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Override batch size from config"
    )
    args = parser.parse_args()

    # Resolve partition path
    if args.partition_path:
        partition_path = args.partition_path
    else:
        partition_path = (
            f"data/partitions/partition_seed42_alpha{args.alpha}_clients{args.num_clients}.json"
        )

    if not Path(partition_path).exists():
        print(f"ERROR: Partition file not found: {partition_path}")
        print("Run `python -m experiments.run_partition` first to generate partitions.")
        sys.exit(1)

    print(f"[Client {args.client_id}] Loading partition from {partition_path}")

    # Load dataset and partition
    train_dataset, _, test_dataset, num_classes = get_bloodmnist_datasets(
        download=True
    )
    client_indices = load_partition(partition_path)

    if args.client_id >= len(client_indices):
        print(
            f"ERROR: client_id={args.client_id} but partition only has "
            f"{len(client_indices)} clients (0-{len(client_indices)-1})"
        )
        sys.exit(1)

    my_indices = client_indices[args.client_id]
    print(f"[Client {args.client_id}] Assigned {len(my_indices)} training samples")

    # Create data loaders
    train_loader = get_client_dataloader(
        train_dataset,
        my_indices,
        client_id=args.client_id,
        batch_size=args.batch_size,
        shuffle=True,
    )

    from torch.utils.data import DataLoader

    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=0
    )

    # Create client and start
    client = FedMedAIClient(
        client_id=args.client_id,
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        local_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
    )

    print(f"[Client {args.client_id}] Connecting to {args.server_address}...")
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
    )


if __name__ == "__main__":
    main()
