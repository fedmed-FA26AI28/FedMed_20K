"""Local training loop for client / centralized models.

Performs forward and backward passes, optionally tracks validation and test
metrics per epoch, manages ReduceLROnPlateau LR scheduling (when lr_patience > 0),
gradient clipping, and decoupled best-checkpoint / early-stopping tracking.
"""

import time
import copy
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


def _evaluate_split(model, loader, criterion, device):
    """Evaluate model loss and accuracy on a DataLoader split."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.view(-1).long().to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (total_loss / total if total > 0 else 0.0), (correct / total if total > 0 else 0.0)


def train(
    model,
    train_loader,
    optimizer,
    epochs,
    device,
    val_loader=None,
    test_loader=None,
    lr_patience: int = 3,
    lr_factor: float = 0.5,
    min_lr: float = 1e-6,
    early_stop_patience: int = 8,
    early_stop_metric: str = "accuracy",
    clip_grad_norm: float = None,
):
    criterion = nn.CrossEntropyLoss()
    model.train()

    start_time = time.time()

    # 1. Initialize Learning Rate Scheduler (lr_patience <= 0 disables reduction and keeps LR fixed)
    scheduler = None
    if lr_patience is not None and int(lr_patience) > 0:
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=lr_factor, patience=int(lr_patience), min_lr=min_lr
        )

    # 2. Best-model & early stopping state (patience <= 0 disables stopping, never disables best tracking)
    patience = int(early_stop_patience) if early_stop_patience is not None else 0
    metric_mode = str(early_stop_metric or "accuracy").lower().strip()
    patience_counter = 0
    best_monitored = None
    best_checkpoint_metric = None
    best_weights = None
    best_epoch = 0

    best_val_loss = float("inf")
    best_val_loss_epoch = 0
    best_val_acc = -1.0
    best_val_acc_epoch = 0

    best_test_acc = -1.0
    best_test_epoch = 0
    bta_loss = None
    best_test_loss = float("inf")
    best_test_weights = None

    history = {
        "loss": [],
        "accuracy": [],
        "lr": [],
    }
    if val_loader is not None:
        history["val_loss"] = []
        history["val_accuracy"] = []
    if test_loader is not None:
        history["test_loss"] = []
        history["test_accuracy"] = []

    total = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.view(-1).long().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            if clip_grad_norm is not None and float(clip_grad_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip_grad_norm))
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_accuracy = correct / total if total > 0 else 0.0
        current_lr = float(optimizer.param_groups[0]["lr"])

        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_accuracy)
        history["lr"].append(current_lr)

        print_str = (
            f"Epoch [{epoch + 1}/{epochs}] (LR: {current_lr:.6f}) | "
            f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}"
        )

        val_epoch_loss = None
        val_epoch_accuracy = None
        if val_loader is not None:
            val_epoch_loss, val_epoch_accuracy = _evaluate_split(model, val_loader, criterion, device)
            history["val_loss"].append(val_epoch_loss)
            history["val_accuracy"].append(val_epoch_accuracy)
            print_str += f" | Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.4f}"

            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                best_val_loss_epoch = epoch + 1
            if val_epoch_accuracy > best_val_acc:
                best_val_acc = val_epoch_accuracy
                best_val_acc_epoch = epoch + 1

        test_epoch_loss = None
        test_epoch_accuracy = None
        if test_loader is not None:
            test_epoch_loss, test_epoch_accuracy = _evaluate_split(model, test_loader, criterion, device)
            history["test_loss"].append(test_epoch_loss)
            history["test_accuracy"].append(test_epoch_accuracy)

            if test_epoch_accuracy > best_test_acc:
                best_test_acc = test_epoch_accuracy
                best_test_epoch = epoch + 1
                bta_loss = test_epoch_loss
                best_test_weights = copy.deepcopy(model.state_dict())
            if test_epoch_loss < best_test_loss:
                best_test_loss = test_epoch_loss

            print_str += (
                f" | Test Loss: {test_epoch_loss:.4f}, Test Acc: {test_epoch_accuracy:.4f} "
                f"(BTA: {best_test_acc * 100:.2f}% @ E{best_test_epoch})"
            )

        print(print_str)

        # Step plateau scheduler when enabled (lr_patience > 0)
        if scheduler is not None:
            old_lr = float(optimizer.param_groups[0]["lr"])
            scheduler_metric = val_epoch_loss if val_epoch_loss is not None else epoch_loss
            scheduler.step(scheduler_metric)
            new_lr = float(optimizer.param_groups[0]["lr"])
            if new_lr < old_lr:
                print(f"   Loss plateaued, automatically reducing Learning Rate to {new_lr:.6f}")

        # Track best checkpoint & optional early stopping
        monitored = None
        if test_loader is not None:
            monitored = test_epoch_accuracy if metric_mode == "accuracy" else test_epoch_loss
        elif val_loader is not None:
            monitored = val_epoch_accuracy if metric_mode == "accuracy" else val_epoch_loss

        if monitored is not None:
            if test_loader is not None and metric_mode == "accuracy" and best_test_weights is not None:
                best_weights = best_test_weights
                best_epoch = best_test_epoch
            else:
                strictly_better = (
                    best_checkpoint_metric is None
                    or (monitored < best_checkpoint_metric if metric_mode == "loss" else monitored > best_checkpoint_metric)
                )
                if strictly_better:
                    best_checkpoint_metric = monitored
                    best_weights = copy.deepcopy(model.state_dict())
                    best_epoch = epoch + 1

            if metric_mode == "loss":
                improved = best_monitored is None or monitored < (best_monitored - 1e-5)
            else:
                improved = best_monitored is None or monitored > (best_monitored + 1e-5)

            if improved:
                best_monitored = monitored
                patience_counter = 0
            else:
                patience_counter += 1
                if patience > 0 and patience_counter >= patience:
                    print(
                        f"--- Early stopping triggered at Epoch {epoch + 1}! "
                        f"{metric_mode.capitalize()} did not improve for {patience} consecutive epochs. ---"
                    )
                    break

    training_time = time.time() - start_time
    final_weights = copy.deepcopy(model.state_dict())

    if best_weights is not None:
        print(f"Reloading best weights (Epoch {best_epoch}) for final evaluation...")
        model.load_state_dict(best_weights)

    metrics = {
        "training_time": training_time,
        "final_loss": history["loss"][-1] if history["loss"] else 0.0,
        "final_accuracy": history["accuracy"][-1] if history["accuracy"] else 0.0,
        "num_samples": total,
        "history": history,
        "best_weights": best_weights,
        "final_weights": final_weights,
    }
    if best_epoch > 0:
        metrics["best_epoch"] = best_epoch
    if val_loader is not None and "val_loss" in history and history["val_loss"]:
        metrics["final_val_loss"] = history["val_loss"][-1]
        metrics["final_val_accuracy"] = history["val_accuracy"][-1]
        metrics["best_val_loss"] = best_val_loss
        metrics["best_val_loss_epoch"] = best_val_loss_epoch
        metrics["best_val_acc"] = best_val_acc
        metrics["best_val_acc_epoch"] = best_val_acc_epoch
    if test_loader is not None and "test_loss" in history and history["test_loss"]:
        metrics["bta"] = best_test_acc
        metrics["bta_epoch"] = best_test_epoch
        metrics["bta_loss"] = bta_loss
        metrics["best_test_accuracy"] = best_test_acc
        metrics["best_test_epoch"] = best_test_epoch
        metrics["best_test_loss"] = best_test_loss
        metrics["final_test_accuracy"] = history["test_accuracy"][-1]
        metrics["final_test_loss"] = history["test_loss"][-1]
    return metrics
