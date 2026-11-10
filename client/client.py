"""FL Client entry point.

Implements a Flower NumPyClient that:
- Loads a local data partition (from a pre-generated Dirichlet JSON).
- Trains the CNN model locally with LR scheduling and early stopping.
- Reports extended metrics: per-epoch time, total training time, weight size.
- Supports FedProx proximal term when instructed by the server config.
- Supports FedSLD shared label distribution reweighted loss.

Usage:
    python -m client.client --client_id 0 --server_address 127.0.0.1:8080
    python -m client.client --client_id 5 --server_address 192.168.1.15:8080 --partition_path data/partitions/partition_seed42_alpha0.3_clients10.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import flwr as fl
from flwr.common import NDArrays, Scalar

from models import (
    get_model,
    set_seed,
    get_parameters,
    set_parameters,
    set_parameters_fedbn,
    is_bn_key,
)
from algorithms.scr import classifier_head, client_support_parameters, head_penalty
from datasets.medmnist_code import get_medmnist_datasets
from datasets.partition import load_partition, get_client_dataloader, get_partition_path
from utils.path_utils import sanitize_run_name


from monitoring.resource import get_resource_usage, get_device_type
from monitoring.network import ClientPingLogger, get_device_ipv4


# ──────────────────────────────────────────────────────────────
# FedSOL Adversarial Proximal Perturbation Optimizer (CVPR 2024)
# ──────────────────────────────────────────────────────────────

class ExpSAM(torch.optim.Optimizer):
    """Adversarial Proximal Perturbation Optimizer for FedSOL (CVPR 2024).

    Wraps a base optimizer (SGD, Adam, AdamW) to perform:
    1. first_step(): Perturbs target parameters (e.g., classifier head or all layers)
                     along the proximal gradient direction.
    2. second_step(): Restores perturbed parameters and steps base optimizer across
                      all trainable parameters using local loss gradients.
    """

    def __init__(
        self,
        params,
        base_optimizer: torch.optim.Optimizer,
        rho: float = 2.0,
        adaptive: bool = False,
        ref_params: Optional[List[torch.Tensor]] = None,
    ):
        target_list = list(params)
        defaults = dict(rho=float(rho), adaptive=bool(adaptive))
        super().__init__(target_list, defaults)
        self.base_optimizer = base_optimizer
        self.target_params = target_list
        self.ref_params = list(ref_params) if ref_params is not None else None

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self._grad_norm()
        scale = self.defaults["rho"] / (grad_norm + 1e-12)
        adaptive = self.defaults.get("adaptive", False)

        for idx, p in enumerate(self.target_params):
            if p.grad is None:
                continue
            self.state[p]["old_p"] = p.data.clone()
            if adaptive and self.ref_params is not None and idx < len(self.ref_params):
                diff = torch.abs(p - self.ref_params[idx])
                diff_norm = torch.norm(diff, p=2) + 1e-12
                lambda_vec = diff / diff_norm
                e_w = lambda_vec * p.grad * scale.to(p)
            else:
                e_w = p.grad * scale.to(p)
            p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for p in self.target_params:
            if p.grad is None or "old_p" not in self.state[p]:
                continue
            p.data.copy_(self.state[p]["old_p"])

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()
            self.base_optimizer.zero_grad()

    def _grad_norm(self):
        adaptive = self.defaults.get("adaptive", False)
        grads = []
        for idx, p in enumerate(self.target_params):
            if p.grad is not None:
                if adaptive and self.ref_params is not None and idx < len(self.ref_params):
                    diff = torch.abs(p - self.ref_params[idx])
                    diff_norm = torch.norm(diff, p=2) + 1e-12
                    lambda_vec = diff / diff_norm
                    grads.append((lambda_vec * p.grad).norm(p=2))
                else:
                    grads.append(p.grad.norm(p=2))
        if not grads:
            return torch.tensor(0.0)
        shared_device = self.target_params[0].device
        return torch.norm(torch.stack([g.to(shared_device) for g in grads]), p=2)


def _extract_class_counts(loader, num_classes: int) -> dict[str, int]:
    """Extract class occurrence frequencies from a DataLoader."""
    counts = np.zeros(num_classes, dtype=np.int64)
    if hasattr(loader.dataset, "labels"):
        lbls = np.asarray(loader.dataset.labels).flatten()
        b = np.bincount(lbls, minlength=num_classes)
        for i in range(min(num_classes, len(b))):
            counts[i] = int(b[i])
    elif hasattr(loader.dataset, "targets"):
        lbls = np.asarray(loader.dataset.targets).flatten()
        b = np.bincount(lbls, minlength=num_classes)
        for i in range(min(num_classes, len(b))):
            counts[i] = int(b[i])
    else:
        for _, labels in loader:
            lbls = labels.view(-1).cpu().numpy()
            for l in lbls:
                if 0 <= l < num_classes:
                    counts[l] += 1
    return {str(c): int(counts[c]) for c in range(num_classes)}


# ──────────────────────────────────────────────────────────────
# Local training function (FL-aware version with proximal term)
# ──────────────────────────────────────────────────────────────

def _local_train(
    model: nn.Module,
    train_loader,
    optimizer,
    device: torch.device,
    epochs: int,
    scheduler=None,
    proximal_mu: float = 0.0,
    global_params: list = None,
    scaffold_drift: list = None,
    is_scaffold: bool = False,
    min_lr: float = 1e-6,
    early_stop_patience: int = 5,
    scr_sample_weights: "torch.Tensor | None" = None,
    scr_anchor: "torch.Tensor | None" = None,
    scr_reference: "tuple[torch.Tensor, torch.Tensor] | None" = None,
    scr_head_lambda: float = 0.0,
    is_fedsld: bool = False,
    fedsld_prior: "torch.Tensor | None" = None,
    fedsld_epsilon: float = 1e-6,
    fedsld_reduction: str = "mean",
    is_fedsol: bool = False,
    fedsol_ref_model: "nn.Module | None" = None,
    fedsol_rho: float = 2.0,
    fedsol_target: str = "head",
    fedsol_loss_type: str = "kl",
    fedsol_temperature: float = 3.0,
    fedsol_adaptive: bool = False,
    clip_grad_norm: "float | None" = None,
):
    """Train locally and return rich metrics.

    Args:
        scheduler: Optional persistent ReduceLROnPlateau scheduler.
        proximal_mu: If > 0 and global_params provided, adds FedProx proximal term.
        global_params: Global model parameters (list of tensors) for proximal term.
        scaffold_drift: List of tensor drift terms (c - c_i) for trainable parameters in SCAFFOLD.
        is_scaffold: Whether SCAFFOLD is active to track uncorrupted loss gradients (Option 2).
        scr_sample_weights: Fixed per-class training-sample weights for SCR.
        scr_anchor: Support-dependent classifier row anchor for SCR.
        scr_reference: Received global classifier weight and bias at round start.
        scr_head_lambda: SCR head anchor penalty strength.
        is_fedsld: Whether FedSLD batch reweighting is active.
        fedsld_prior: Global prior label distribution P~(y) tensor of shape (C,).
        fedsld_epsilon: Smoothing epsilon to prevent zero division in FedSLD.
        fedsld_reduction: Batch loss reduction ('mean' or 'sum').
        is_fedsol: Whether FedSOL adversarial proximal perturbation is active.
        fedsol_ref_model: Reference global model for FedSOL proximal discrepancy.
        fedsol_rho: Perturbation radius strength for FedSOL (CVPR 2024 default: 2.0).
        fedsol_target: Perturbation target layer ('head' = classifier head only, 'all' = full model).
        fedsol_loss_type: Proximal discrepancy metric ('kl' or 'l2').
        fedsol_temperature: Softmax temperature for KL divergence.
        fedsol_adaptive: Whether to use adaptive perturbation radius in FedSOL.
        clip_grad_norm: Maximum L2 gradient norm for clipping (None or <= 0 disables).
    """
    criterion = nn.CrossEntropyLoss()
    use_scr = scr_sample_weights is not None
    if use_scr:
        if scr_anchor is None or scr_reference is None:
            raise ValueError("SCR requires sample weights, anchor and global head reference")
        scr_sample_weights = scr_sample_weights.to(device)
        scr_anchor = scr_anchor.to(device)
        scr_reference = tuple(value.to(device) for value in scr_reference)

    if is_fedsld and fedsld_prior is not None:
        fedsld_prior = fedsld_prior.to(device)

    best_loss = float("inf")
    patience_counter = 0
    epoch_times = []
    epoch_losses = []
    epoch_accuracies = []
    epoch_lrs = []
    total_samples = 0
    total_steps = 0

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    grad_accum = [torch.zeros_like(p) for p in trainable_params] if is_scaffold else None

    # Identify the final classifier layer for FedSOL partial perturbation
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m

    head_params = []
    if last_linear is not None:
        head_params = [p for p in last_linear.parameters() if p.requires_grad]

    ref_head_params = []
    if fedsol_ref_model is not None and last_linear is not None:
        ref_last_linear = None
        for m in fedsol_ref_model.modules():
            if isinstance(m, nn.Linear):
                ref_last_linear = m
        if ref_last_linear is not None:
            ref_head_params = [p for p in ref_last_linear.parameters()]

    # FedSOL optimizer wrapper (SAM-based adversarial proximal perturbation)
    sam_optimizer = None
    if is_fedsol and fedsol_ref_model is not None:
        if str(fedsol_target).lower() == "head" and head_params:
            target_p = head_params
            ref_p = ref_head_params if ref_head_params else [p for p in fedsol_ref_model.parameters()]
        else:
            target_p = trainable_params
            ref_p = [p for p in fedsol_ref_model.parameters()]

        sam_optimizer = ExpSAM(
            params=target_p,
            base_optimizer=optimizer,
            rho=fedsol_rho,
            adaptive=fedsol_adaptive,
            ref_params=ref_p,
        )

    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.view(-1).long().to(device)

            # ── Special Branch: FedSOL Two-Pass Adversarial Proximal Update ──
            if is_fedsol and fedsol_ref_model is not None and sam_optimizer is not None:
                # Pass 1: Measure discrepancy to global reference model and perturb
                optimizer.zero_grad()
                local_logits = model(images)
                with torch.no_grad():
                    ref_logits = fedsol_ref_model(images)

                if fedsol_loss_type == "kl":
                    T = max(fedsol_temperature, 1e-4)
                    pred_probs = F.log_softmax(local_logits / T, dim=1)
                    target_probs = F.softmax(ref_logits / T, dim=1)
                    prox_loss = F.kl_div(pred_probs, target_probs, reduction="batchmean") * (T * T)
                else:  # 'l2' parameter distance
                    prox_loss = 0.0
                    for lp, rp in zip(model.parameters(), fedsol_ref_model.parameters()):
                        prox_loss = prox_loss + 0.5 * ((lp - rp) ** 2).sum()

                prox_loss.backward()
                sam_optimizer.first_step(zero_grad=True)

                # Pass 2: Evaluate local task loss on perturbed weights & update
                optimizer.zero_grad()
                perturbed_logits = model(images)
                loss = criterion(perturbed_logits, labels)
                loss.backward()

                if clip_grad_norm is not None and clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=clip_grad_norm)

                sam_optimizer.second_step(zero_grad=True)

                total_steps += 1
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(perturbed_logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                continue

            optimizer.zero_grad()
            outputs = model(images)

            # Strategy-specific loss formulation
            if is_fedsld and fedsld_prior is not None:
                # FedSLD: reweight local mini-batch CE by p_b(y) / (P~(y) + eps)
                batch_counts = torch.bincount(labels, minlength=fedsld_prior.size(0)).float()
                pb = batch_counts / float(labels.size(0))
                weights = pb[labels] / (fedsld_prior[labels] + fedsld_epsilon)
                # Self-normalize weights to unit mean across batch to preserve standard CE gradient scale
                # and prevent exponential divergence under severe non-IID label skew
                weights = weights / (weights.mean() + 1e-8)
                ce = F.cross_entropy(outputs, labels, reduction="none")
                if fedsld_reduction == "sum":
                    loss = (weights * ce).sum()
                else:
                    loss = (weights * ce).mean()
            elif use_scr:
                ce = F.cross_entropy(outputs, labels, reduction="none")
                loss = (scr_sample_weights[labels] * ce).mean()
                loss = loss + head_penalty(
                    model, scr_reference, scr_anchor, scr_head_lambda)
            else:
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

            # SCAFFOLD Option 2: Accumulate uncorrupted loss gradient before adding drift
            if grad_accum is not None:
                for accum, p in zip(grad_accum, trainable_params):
                    if p.grad is not None:
                        accum.add_(p.grad)

            if clip_grad_norm is not None and clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=clip_grad_norm)

            optimizer.step()

            # SCAFFOLD drift correction (NVFlare decoupled post-step):
            if scaffold_drift is not None:
                step_lr = float(optimizer.param_groups[0]["lr"])
                with torch.no_grad():
                    for p, drift in zip(trainable_params, scaffold_drift):
                        p.data.add_(drift, alpha=-step_lr)
            total_steps += 1

            running_loss += loss.item() * images.size(0)
            # Accuracy always computed from raw logits
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_times.append(epoch_time)
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_acc)
        epoch_lrs.append(current_lr)
        total_samples = total

        print(
            f"  [Client] Epoch [{epoch+1}/{epochs}] "
            f"LR={current_lr:.6f} | Loss={epoch_loss:.4f} Acc={epoch_acc:.4f} "
            f"| {epoch_time:.2f}s"
        )

        # LR scheduling on client's own train loss
        old_lr = current_lr
        if scheduler is not None:
            scheduler.step(epoch_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < old_lr:
            print(f"    -> [Client LR Reducer] Loss plateaued: LR reduced {old_lr:.6f} -> {new_lr:.6f}")

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

    metrics_out = {
        "train_loss": epoch_loss,
        "train_accuracy": epoch_acc,
        "epoch_times": epoch_times,
        "epoch_losses": epoch_losses,
        "epoch_accuracies": epoch_accuracies,
        "epoch_lrs": epoch_lrs,
        "epoch_time_avg": float(np.mean(epoch_times)),
        "epoch_time_min": float(np.min(epoch_times)),
        "epoch_time_max": float(np.max(epoch_times)),
        "num_samples": total_samples,
        "epochs_run": len(epoch_times),
        "total_steps": total_steps,
    }
    if grad_accum is not None:
        metrics_out["grad_accum"] = [
            (accum / max(1, total_steps)).cpu().numpy().astype(np.float64)
            for accum in grad_accum
        ]
    return metrics_out


# ──────────────────────────────────────────────────────────────
# Flower NumPyClient
# ──────────────────────────────────────────────────────────────

def _training_class_counts(train_loader, num_classes: int) -> np.ndarray:
    """Count only local training labels, without loading augmented images."""
    subset = train_loader.dataset
    source = getattr(subset, "dataset", None)
    indices = getattr(subset, "indices", None)
    if source is not None and indices is not None and hasattr(source, "labels"):
        labels = np.asarray(source.labels).reshape(-1)[np.asarray(indices, dtype=int)]
    elif hasattr(subset, "labels"):
        labels = np.asarray(subset.labels).reshape(-1)
    else:
        labels = np.concatenate([
            batch_labels.reshape(-1).cpu().numpy() for _, batch_labels in train_loader
        ])
    if len(labels) != len(subset) or np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("SCR local training labels do not match the model classes")
    return np.bincount(labels.astype(np.int64), minlength=num_classes)


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
        lr_mode: str = "client_loss",
        lr_patience: int = 2,
        lr_factor: float = 0.5,
        lr_min: float = 1e-6,
        device_type: str = None,
        save_local_metrics: bool = True,
        server_address: str = "127.0.0.1:8080",
        strategy: str = None,
        partition_path: Optional[str] = None,
        client_dir: Optional[Union[str, Path]] = None,
        model_name: str = "cnn",
        optimizer_name: str = "adam",
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        name: Optional[str] = None,
        fedsol_rho: float = 2.0,
        fedsol_target: str = "head",
        fedsol_loss_type: str = "kl",
        fedsol_temperature: float = 3.0,
        fedsol_adaptive: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        self.client_id = client_id
        self.model_name = model_name
        self.num_classes = num_classes
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.optimizer_name = str(optimizer_name).lower().strip() if optimizer_name else "adam"
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.lr_mode = str(lr_mode).lower().strip() if lr_mode else "client_loss"
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_min = lr_min
        self._lr_initialized = False
        self.device_type = str(device_type).strip() if device_type else get_device_type(client_id)
        self.save_local_metrics = save_local_metrics
        self.server_address = server_address
        self.strategy = strategy.lower() if strategy else None
        self.partition_path = str(partition_path) if partition_path else None
        self.client_control_variate: Optional[List[np.ndarray]] = None

        self.fedsol_rho = float(fedsol_rho)
        self.fedsol_target = str(fedsol_target).lower().strip()
        self.fedsol_loss_type = str(fedsol_loss_type).lower().strip()
        self.fedsol_temperature = float(fedsol_temperature)
        self.fedsol_adaptive = bool(fedsol_adaptive)
        self.clip_grad_norm = float(clip_grad_norm) if clip_grad_norm is not None and clip_grad_norm > 0 else None

        self._class_counts_dict = _extract_class_counts(self.train_loader, self.num_classes)
        self._class_counts = torch.tensor(
            [self._class_counts_dict[str(c)] for c in range(self.num_classes)],
            dtype=torch.float32,
        )

        self.round_history = []
        self.epoch_history = []

        # Client ping logger
        if client_dir is not None:
            self.client_dir = Path(client_dir)
        elif name:
            safe_name = sanitize_run_name(name)
            self.client_dir = Path("results") / "clients" / safe_name / f"client_{self.client_id}"
        else:
            self.client_dir = Path("results") / "clients" / f"client_{self.client_id}"
        self.ping_logger = ClientPingLogger(
            client_id=self.client_id,
            server_address=self.server_address,
            device_type=self.device_type,
            log_dir=self.client_dir,
            enabled=self.save_local_metrics,
        )
        self.ping_logger.log_ping(server_round=0, event="init")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = get_model(model_name, num_classes=num_classes).to(self.device)

        if self.optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=self.lr_min,
        )

        self.device_ipv4 = get_device_ipv4()

        print(
            f"[Client {self.client_id}] Initialized on {self.device} ({self.device_type}) | "
            f"IPv4: {self.device_ipv4} | train_samples={len(train_loader.dataset)} | "
            f"Optimizer: {self.optimizer_name.upper()} | LR Mode: {self.lr_mode} (Initial LR: {self.learning_rate})"
        )

    def get_parameters(self, config) -> NDArrays:
        """Return model parameters as numpy arrays."""
        return get_parameters(self.model)

    def fit(self, parameters: NDArrays, config: dict):
        """Receive global model, train locally, return updated weights + metrics."""
        strategy = str(config.get("strategy", self.strategy or "fedavg")).lower()
        num_model_layers = len(get_parameters(self.model))
        fedbn_strategies = (
            "fedbn", "proposed", "fedbns", "fedbn_scaffold", "bn_scaffold", "scaffold_fedbn",
        )
        scaffold_strategies = (
            "scaffold", "proposed", "fedbns", "fedbn_scaffold", "bn_scaffold", "scaffold_fedbn",
        )
        is_scaffold = (strategy in scaffold_strategies) or (len(parameters) == 2 * num_model_layers)
        is_fedbn = strategy in fedbn_strategies

        initial_weights = None
        server_c = None
        scaffold_drift = None

        if is_scaffold:
            if len(parameters) == 2 * num_model_layers:
                initial_weights = parameters[:num_model_layers]
                server_c = parameters[num_model_layers:]
            else:
                initial_weights = parameters
                server_c = [np.zeros_like(p) for p in initial_weights]

            if is_fedbn:
                set_parameters_fedbn(self.model, initial_weights)
            else:
                set_parameters(self.model, initial_weights)

            if self.client_control_variate is None:
                self.client_control_variate = [np.zeros_like(p) for p in initial_weights]

            # Construct drift correction terms (server_c - client_c) for trainable parameters
            sd_keys = list(self.model.state_dict().keys())
            named_params = dict(self.model.named_parameters())
            scaffold_drift = [
                torch.as_tensor(s - c, device=self.device, dtype=named_params[k].dtype)
                if (not is_fedbn or not is_bn_key(self.model, k))
                else torch.zeros_like(named_params[k])
                for k, s, c in zip(sd_keys, server_c, self.client_control_variate)
                if k in named_params
            ]
        elif is_fedbn:
            set_parameters_fedbn(self.model, parameters)
        else:
            set_parameters(self.model, parameters)

        # Read config from server
        server_round = int(config.get("server_round", len(self.round_history) + 1))
        local_epochs = int(config.get("local_epochs", self.local_epochs))
        proximal_mu = float(config.get("proximal_mu", 0.0))
        lr_mode = str(config.get("lr_mode", self.lr_mode)).lower().strip()

        if "optimizer" in config:
            srv_opt = str(config["optimizer"]).lower().strip()
            if srv_opt != self.optimizer_name:
                self.optimizer_name = srv_opt
                if "momentum" in config:
                    self.momentum = float(config["momentum"])
                if self.optimizer_name == "sgd":
                    self.optimizer = optim.SGD(
                        self.model.parameters(),
                        lr=self.learning_rate,
                        momentum=self.momentum,
                        weight_decay=self.weight_decay,
                    )
                elif self.optimizer_name == "adamw":
                    self.optimizer = optim.AdamW(
                        self.model.parameters(),
                        lr=self.learning_rate,
                        weight_decay=self.weight_decay,
                    )
                else:
                    self.optimizer = optim.Adam(
                        self.model.parameters(),
                        lr=self.learning_rate,
                        weight_decay=self.weight_decay,
                    )
                print(f"[Client {self.client_id}] Switched local optimizer to {self.optimizer_name.upper()} based on server config")

        active_scheduler = None

        if lr_mode == "fixed":
            # Fixed constant LR mode
            fixed_lr = float(config.get("learning_rate", self.learning_rate))
            for pg in self.optimizer.param_groups:
                pg["lr"] = fixed_lr
            self.learning_rate = fixed_lr
            active_scheduler = None
            print(f"[Client {self.client_id}] Round {server_round}: Fixed LR = {fixed_lr:.6f}")

        elif lr_mode in ("server_decay", "server", "step"):
            # Server-synchronized global step decay
            scheduled_lr = float(config.get("learning_rate", self.learning_rate))
            for pg in self.optimizer.param_groups:
                pg["lr"] = scheduled_lr
            self.learning_rate = scheduled_lr
            active_scheduler = None
            print(f"[Client {self.client_id}] Round {server_round}: Server decayed LR = {scheduled_lr:.6f}")

        else:
            # 'client_loss' mode
            if server_round == 1 or not self._lr_initialized:
                self._lr_initialized = True
                init_lr = config.get("learning_rate")
                if init_lr is None:
                    init_lr = config.get("initial_lr", self.learning_rate)
                init_lr = float(init_lr)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = init_lr
                self.learning_rate = init_lr

                if "client_lr_patience" in config:
                    self.lr_patience = int(config["client_lr_patience"])
                if "client_lr_factor" in config:
                    self.lr_factor = float(config["client_lr_factor"])
                if "client_lr_min" in config:
                    self.lr_min = float(config["client_lr_min"])

                self.scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode="min",
                    factor=self.lr_factor,
                    patience=self.lr_patience,
                    min_lr=self.lr_min,
                )
                print(
                    f"[Client {self.client_id}] Training start (Round {server_round}): Initial LR = {init_lr:.6f} "
                    f"(local reducer: patience={self.lr_patience}, factor={self.lr_factor})"
                )
            else:
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"[Client {self.client_id}] Round {server_round}: Continuing with local LR = {current_lr:.6f} "
                    f"(managed independently by local loss)"
                )
            active_scheduler = self.scheduler

        # Measure network ping to server for this round
        ping_res = self.ping_logger.log_ping(server_round=server_round, event="fit")
        ping_ms = ping_res.get("ping_ms")

        # Prepare global params for FedProx proximal term
        global_params = None
        if proximal_mu > 0.0:
            global_params = [
                p.clone().detach() for p in self.model.parameters()
            ]

        # Prepare SCR parameters
        scr_sample_weights = None
        scr_anchor = None
        scr_reference = None
        scr_head_lambda = 0.0

        if strategy == "scr":
            required = ("scr_global_counts", "scr_coverage_sum", "scr_total_samples",
                        "scr_num_clients", "scr_gamma", "scr_head_lambda")
            missing = [key for key in required if key not in config]
            if missing:
                raise ValueError(f"SCR server configuration missing: {missing}")
            if not hasattr(self, "_scr_class_counts"):
                self._scr_class_counts = _training_class_counts(
                    self.train_loader, self.num_classes)
            local_counts = self._scr_class_counts
            global_counts = np.asarray(json.loads(config["scr_global_counts"]), dtype=np.float64)
            coverage_sum = np.asarray(json.loads(config["scr_coverage_sum"]), dtype=np.float64)
            scr_gamma = float(config["scr_gamma"])
            scr_head_lambda = float(config["scr_head_lambda"])
            if not np.isfinite(scr_head_lambda) or scr_head_lambda < 0:
                raise ValueError("SCR head lambda must be finite and nonnegative")
            weights, anchor = client_support_parameters(
                local_counts, global_counts, coverage_sum,
                int(config["scr_total_samples"]), int(config["scr_num_clients"]),
                scr_gamma,
            )
            scr_sample_weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
            scr_anchor = torch.as_tensor(anchor, dtype=torch.float32, device=self.device)
            head = classifier_head(self.model)
            scr_reference = (head.weight.detach().clone(), head.bias.detach().clone())
            print(f"[Client {self.client_id}] SCR: gamma={scr_gamma:g} "
                  f"head_lambda={scr_head_lambda:g} counts={local_counts.tolist()}")

        # Prepare FedSLD parameters
        fedsld_prior_tensor = None
        is_fedsld = strategy in ("fedsld", "fed_sld")
        fedsld_epsilon = float(config.get("fedsld_epsilon", 1e-6))
        fedsld_reduction = str(config.get("fedsld_reduction", "mean")).lower()

        if is_fedsld:
            if "fedsld_prior" in config:
                prior_list = json.loads(config["fedsld_prior"])
                fedsld_prior_tensor = torch.tensor(prior_list, dtype=torch.float32, device=self.device)
            elif hasattr(self, "partition_path") and self.partition_path and Path(self.partition_path).exists():
                from algorithms.fedsld import load_prior_from_partition
                prior_list = load_prior_from_partition(self.partition_path, num_classes=self.num_classes)
                fedsld_prior_tensor = torch.tensor(prior_list, dtype=torch.float32, device=self.device)
            else:
                fedsld_prior_tensor = torch.ones(self.num_classes, dtype=torch.float32, device=self.device) / float(self.num_classes)

            print(
                f"[Client {self.client_id}] FedSLD: eps={fedsld_epsilon} reduction={fedsld_reduction} "
                f"| prior={[round(p, 4) for p in fedsld_prior_tensor.tolist()]}"
            )

        # Prepare FedSOL parameters
        is_fedsol = strategy in ("fedsol", "fed_sol")
        fedsol_rho = float(config.get("fedsol_rho", self.fedsol_rho))
        fedsol_target = str(config.get("fedsol_target", getattr(self, "fedsol_target", "head"))).lower().strip()
        fedsol_loss_type = str(config.get("fedsol_loss_type", self.fedsol_loss_type)).lower()
        fedsol_temperature = float(config.get("fedsol_temperature", self.fedsol_temperature))
        fedsol_adaptive = bool(config.get("fedsol_adaptive", self.fedsol_adaptive))
        clip_grad_norm_val = config.get("clip_grad_norm", self.clip_grad_norm)
        if clip_grad_norm_val is not None:
            clip_grad_norm_val = float(clip_grad_norm_val) if float(clip_grad_norm_val) > 0 else None

        fedsol_ref_model = None
        if is_fedsol:
            import copy
            fedsol_ref_model = copy.deepcopy(self.model)
            fedsol_ref_model.eval()
            for p in fedsol_ref_model.parameters():
                p.requires_grad = False
            print(
                f"[Client {self.client_id}] FedSOL: rho={fedsol_rho} target={fedsol_target} "
                f"loss_type={fedsol_loss_type} T={fedsol_temperature} adaptive={fedsol_adaptive}"
            )

        # Train locally
        train_start = time.time()
        train_metrics = _local_train(
            model=self.model,
            train_loader=self.train_loader,
            optimizer=self.optimizer,
            device=self.device,
            epochs=local_epochs,
            scheduler=active_scheduler,
            proximal_mu=proximal_mu,
            global_params=global_params,
            scaffold_drift=scaffold_drift,
            is_scaffold=is_scaffold,
            scr_sample_weights=scr_sample_weights,
            scr_anchor=scr_anchor,
            scr_reference=scr_reference,
            scr_head_lambda=scr_head_lambda,
            is_fedsld=is_fedsld,
            fedsld_prior=fedsld_prior_tensor,
            fedsld_epsilon=fedsld_epsilon,
            fedsld_reduction=fedsld_reduction,
            is_fedsol=is_fedsol,
            fedsol_ref_model=fedsol_ref_model,
            fedsol_rho=fedsol_rho,
            fedsol_target=fedsol_target,
            fedsol_loss_type=fedsol_loss_type,
            fedsol_temperature=fedsol_temperature,
            fedsol_adaptive=fedsol_adaptive,
            clip_grad_norm=clip_grad_norm_val,
        )
        training_time = time.time() - train_start

        # Compute updated parameters
        updated_model_params = get_parameters(self.model)

        if is_scaffold:
            K = max(int(train_metrics.get("total_steps", 1)), 1)
            sd_keys = list(self.model.state_dict().keys())
            named_params = dict(self.model.named_parameters())
            grad_avg = train_metrics.get("grad_accum")

            delta_c = []
            new_client_c = []
            trainable_idx = 0
            for idx, k in enumerate(sd_keys):
                if k in named_params:
                    is_bn = is_fedbn and is_bn_key(self.model, k)
                    if not is_bn:
                        if grad_avg is not None and trainable_idx < len(grad_avg):
                            c_i_new = grad_avg[trainable_idx]
                            d_c = c_i_new - self.client_control_variate[idx]
                        else:
                            lr = float(self.optimizer.param_groups[0]["lr"])
                            d_c = (1.0 / (K * lr)) * (initial_weights[idx] - updated_model_params[idx]) - server_c[idx]
                            c_i_new = self.client_control_variate[idx] + d_c
                    else:
                        d_c = np.zeros_like(updated_model_params[idx], dtype=np.float64)
                        c_i_new = np.zeros_like(updated_model_params[idx], dtype=np.float64)
                    trainable_idx += 1
                else:
                    d_c = np.zeros_like(updated_model_params[idx], dtype=np.float64)
                    c_i_new = np.zeros_like(updated_model_params[idx], dtype=np.float64)
                delta_c.append(d_c)
                new_client_c.append(c_i_new)

            self.client_control_variate = new_client_c
            return_parameters = updated_model_params + delta_c
        else:
            return_parameters = updated_model_params

        # Sample hardware resources
        res = get_resource_usage()

        # Compute weight size in KB
        weight_size_bytes = sum(p.nbytes for p in return_parameters)
        weight_size_kb = weight_size_bytes / 1024.0

        # Build metrics dict
        metrics = {
            "client_id": int(self.client_id),
            "device_type": str(self.device_type),
            "device_ipv4": str(self.device_ipv4),
            "server_round": int(server_round),
            "num_samples": int(train_metrics["num_samples"]),
            "local_epochs": int(local_epochs),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_metrics["train_loss"]),
            "train_accuracy": float(train_metrics["train_accuracy"]),
            "training_time": float(training_time),
            "epoch_time_avg": float(train_metrics["epoch_time_avg"]),
            "epoch_time_min": float(train_metrics["epoch_time_min"]),
            "epoch_time_max": float(train_metrics["epoch_time_max"]),
            "epoch_times": json.dumps([round(t, 4) for t in train_metrics["epoch_times"]]),
            "epoch_losses": json.dumps([round(l, 4) for l in train_metrics["epoch_losses"]]),
            "epoch_accuracies": json.dumps([round(a, 4) for a in train_metrics["epoch_accuracies"]]),
            "epoch_lrs": json.dumps([round(lr, 6) for lr in train_metrics["epoch_lrs"]]),
            "epochs_run": int(train_metrics["epochs_run"]),
            "local_steps": int(train_metrics.get("total_steps", train_metrics["epochs_run"])),
            "ping_ms": float(ping_ms) if ping_ms is not None else -1.0,
            "weight_size_kb": float(weight_size_kb),
            "cpu_percent": float(res["cpu_percent"]),
            "ram_percent": float(res["ram_percent"]),
            "ram_used_mb": float(res["ram_used_mb"]),
            "gpu_memory_mb": float(res["gpu_memory_allocated_mb"]),
        }
        metrics["class_counts_json"] = json.dumps(self._class_counts_dict)
        if is_fedsol:
            metrics["fedsol_active"] = True
            metrics["fedsol_rho"] = fedsol_rho
            metrics["fedsol_target"] = fedsol_target
            metrics["fedsol_loss_type"] = fedsol_loss_type
        if strategy == "scr":
            metrics["scr_active"] = True
            metrics["scr_gamma"] = float(config["scr_gamma"])
            metrics["scr_head_lambda"] = scr_head_lambda
        if is_fedsld:
            metrics["fedsld_active"] = True
            metrics["fedsld_epsilon"] = fedsld_epsilon
            metrics["fedsld_reduction"] = fedsld_reduction

        # Track history locally
        self.round_history.append(metrics)
        cum_t = 0.0
        for ep_idx, ep_time in enumerate(train_metrics["epoch_times"]):
            cum_t += ep_time
            self.epoch_history.append({
                "round": server_round,
                "client_id": self.client_id,
                "device_type": self.device_type,
                "device_ipv4": self.device_ipv4,
                "epoch": ep_idx + 1,
                "epoch_time_seconds": round(ep_time, 4),
                "cumulative_epoch_time_seconds": round(cum_t, 4),
                "train_loss": round(train_metrics["epoch_losses"][ep_idx], 6),
                "train_accuracy": round(train_metrics["epoch_accuracies"][ep_idx], 6),
                "learning_rate": train_metrics["epoch_lrs"][ep_idx],
            })

        if self.save_local_metrics:
            self._save_client_csvs()

        print(
            f"[Client {self.client_id}] fit done | Round {server_round} | IPv4: {self.device_ipv4} | "
            f"loss={metrics['train_loss']:.4f} acc={metrics['train_accuracy']:.4f} "
            f"| round_time={training_time:.2f}s | avg_epoch={metrics['epoch_time_avg']:.2f}s "
            f"| weights={weight_size_kb:.1f}KB"
        )

        return return_parameters, train_metrics["num_samples"], metrics

    def _save_client_csvs(self):
        """Export local client CSV metrics for on-device inspection."""
        try:
            client_dir = self.client_dir
            client_dir.mkdir(parents=True, exist_ok=True)

            # Round metrics
            r_path = client_dir / f"client_{self.client_id}_round_metrics.csv"
            r_cols = [
                "server_round", "client_id", "device_type", "num_samples", "local_epochs",
                "learning_rate", "training_time", "epoch_time_avg", "epoch_time_min",
                "epoch_time_max", "train_loss", "train_accuracy", "ping_ms", "weight_size_kb",
                "cpu_percent", "ram_percent", "ram_used_mb", "gpu_memory_mb"
            ]
            import csv
            with open(r_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=r_cols, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.round_history)

            # Epoch metrics
            ep_path = client_dir / f"client_{self.client_id}_epoch_metrics.csv"
            ep_cols = [
                "round", "client_id", "device_type", "epoch", "epoch_time_seconds",
                "cumulative_epoch_time_seconds", "train_loss", "train_accuracy", "learning_rate"
            ]
            with open(ep_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=ep_cols, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.epoch_history)
        except Exception as e:
            print(f"[Client {self.client_id}] Warning: could not write local CSV: {e}")

    def evaluate(self, parameters: NDArrays, config: dict):
        """Evaluate the global model on the local test set."""
        eval_round = config.get("server_round")
        if eval_round is not None:
            self.ping_logger.log_ping(server_round=int(eval_round), event="evaluate")

        strategy = str(config.get("strategy", self.strategy or "fedavg")).lower()
        fedbn_eval_strats = (
            "fedbn", "proposed", "fedbns", "fedbn_scaffold", "bn_scaffold", "scaffold_fedbn",
        )
        if strategy in fedbn_eval_strats:
            set_parameters_fedbn(self.model, parameters)
        else:
            set_parameters(self.model, parameters)

        self.model.eval()

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.view(-1).long().to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        if total == 0:
            # A Dirichlet split can give a training client no local test images.
            # Report zero examples so sample-weighted evaluation ignores it.
            if not getattr(self, "_empty_test_warning_issued", False):
                print(f"[Client {self.client_id}] No local test samples; skipping evaluation")
                self._empty_test_warning_issued = True
            return 0.0, 0, {"accuracy": 0.0}

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

def load_yaml_config(config_path: str = "configs/experiment.yaml") -> dict:
    """Load configuration dictionary from YAML file if available."""
    if not os.path.exists(config_path):
        return {}
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg
    except Exception:
        return {}


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

    server_host = yaml_cfg.get("server_host", "127.0.0.1")
    if server_host == "0.0.0.0":
        server_host = "127.0.0.1"
    server_port = yaml_cfg.get("server_port", 8080)
    default_server_address = yaml_cfg.get("server_address") or f"{server_host}:{server_port}"

    active_clients = yaml_cfg.get("active_clients")
    default_num_clients = len(active_clients) if isinstance(active_clients, list) and active_clients else yaml_cfg.get("num_clients", yaml_cfg.get("min_clients", 12))

    parser = argparse.ArgumentParser(
        description="FedMedAI FL Client",
        parents=[pre_parser],
    )
    parser.add_argument(
        "--client_id", type=int, required=True, help="Client ID integer (0, 1, 2, ...)"
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default=default_server_address,
        help=f"FL Server address (default: {default_server_address})",
    )
    parser.add_argument(
        "--partition_path",
        type=str,
        default=None,
        help="Path to partition JSON. If None, auto-detect from data/partitions/",
    )
    parser.add_argument(
        "--dataset", type=str, default=yaml_cfg.get("dataset", "bloodmnist"), help=f"Dataset name (bloodmnist, pathmnist, organamnist, dermamnist) (default: {yaml_cfg.get('dataset', 'bloodmnist')})"
    )
    parser.add_argument(
        "--partition_mode", type=str, default=yaml_cfg.get("partition_mode", yaml_cfg.get("partition", "dirichlet_0.3")), help=f"Partition mode (default: {yaml_cfg.get('partition_mode', yaml_cfg.get('partition', 'dirichlet_0.3'))})"
    )
    parser.add_argument(
        "--num_clients", type=int, default=default_num_clients, help=f"Total number of clients (default: {default_num_clients})"
    )
    parser.add_argument(
        "--local_epochs", type=int, default=yaml_cfg.get("local_epochs", 5), help=f"Local training epochs per round (default: {yaml_cfg.get('local_epochs', 5)})"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=yaml_cfg.get("learning_rate", 0.001), help=f"Initial base learning rate (default: {yaml_cfg.get('learning_rate', 0.001)})"
    )
    parser.add_argument(
        "--lr_patience",
        "--client_lr_patience",
        dest="lr_patience",
        type=int,
        default=yaml_cfg.get("client_lr_patience", 2),
        help=f"Client local epochs without loss improvement before reducing LR (default: {yaml_cfg.get('client_lr_patience', 2)})",
    )
    parser.add_argument(
        "--lr_factor",
        "--client_lr_factor",
        dest="lr_factor",
        type=float,
        default=yaml_cfg.get("client_lr_factor", 0.5),
        help=f"Multiplicative factor for client LR reduction (default: {yaml_cfg.get('client_lr_factor', 0.5)})",
    )
    parser.add_argument(
        "--lr_min",
        "--client_lr_min",
        dest="lr_min",
        type=float,
        default=yaml_cfg.get("client_lr_min", 1e-6),
        help=f"Minimum client learning rate bound (default: {yaml_cfg.get('client_lr_min', 1e-6)})",
    )
    parser.add_argument(
        "--lr_mode",
        type=str,
        default=yaml_cfg.get("lr_mode", "client_loss"),
        choices=["client_loss", "server_decay", "fixed"],
        help=f"LR schedule mode: 'client_loss' (each client reduces LR on local loss plateau), 'server_decay' (step decay on rounds), 'fixed' (constant LR) (default: {yaml_cfg.get('lr_mode', 'client_loss')})",
    )
    parser.add_argument(
        "--batch_size", type=int, default=yaml_cfg.get("batch_size", None), help="Override batch size from config (default: from config or jetson.yaml)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=yaml_cfg.get("num_workers", 0 if sys.platform == "win32" else None),
        help=f"DataLoader worker processes (default: 0 on Windows, or from config)",
    )
    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=yaml_cfg.get("pin_memory", None),
        help="Enable/disable pinned host memory for DataLoader transfers",
    )
    parser.add_argument(
        "--device_type",
        type=str,
        default=None,
        help="Device class/type override (e.g. 'pc', 'jetson_orin', 'jetson_nano', 'PC1'). Defaults to auto-detected from configs/jetson.yaml.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=yaml_cfg.get("strategy", "fedavg"),
        choices=[
            "fedavg",
            "fedprox",
            "fednova",
            "fedbn",
            "scaffold",
            "fedbns",
            "proposed",
            "fedbn_scaffold",
            "scr",
            "fedsld",
            "fed_sld",
            "fedsol",
            "fed_sol",
            "fedswitch",
            "fed_switch",
        ],
        help=f"FL strategy (default: {yaml_cfg.get('strategy', 'fedavg')})",
    )
    parser.add_argument(
        "--client_dir",
        type=str,
        default=None,
        help="Custom directory to store client metrics and ping logs (default: results/clients/[<name>/]client_<id>)",
    )
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=yaml_cfg.get("name", None),
        help="Designated run name for client output directory (default: None)",
    )
    parser.add_argument(
        "--local_test",
        action=argparse.BooleanOptionalAction,
        default=yaml_cfg.get("local_test", False),
        help="Evaluate on client-specific local test partition (following FedSLD paper)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=yaml_cfg.get("model", "cnn"),
        choices=["cnn", "baseline", "paper", "classic_cnn", "baseline_bn", "baselinecnn_bn", "classic_bn"],
        help=f"Model architecture (default: {yaml_cfg.get('model', 'cnn')})",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=yaml_cfg.get("optimizer", "adam"),
        choices=["adam", "sgd", "adamw"],
        help=f"Client local optimizer ('adam', 'sgd', or 'adamw') (default: {yaml_cfg.get('optimizer', 'adam')})",
    )
    parser.add_argument(
        "--fedsol_rho",
        type=float,
        default=yaml_cfg.get("fedsol_rho", 2.0),
        help=f"FedSOL perturbation radius rho (default: {yaml_cfg.get('fedsol_rho', 2.0)})",
    )
    parser.add_argument(
        "--fedsol_target",
        type=str,
        default=yaml_cfg.get("fedsol_target", "head"),
        choices=["head", "all"],
        help=f"FedSOL perturbation target layer ('head' or 'all') (default: {yaml_cfg.get('fedsol_target', 'head')})",
    )
    parser.add_argument(
        "--fedsol_loss_type",
        type=str,
        default=yaml_cfg.get("fedsol_loss_type", "kl"),
        choices=["kl", "l2"],
        help=f"FedSOL proximal discrepancy loss type ('kl' or 'l2') (default: {yaml_cfg.get('fedsol_loss_type', 'kl')})",
    )
    parser.add_argument(
        "--fedsol_temperature",
        type=float,
        default=yaml_cfg.get("fedsol_temperature", 3.0),
        help=f"FedSOL softmax temperature T for KL divergence (default: {yaml_cfg.get('fedsol_temperature', 3.0)})",
    )
    parser.add_argument(
        "--fedsol_adaptive",
        action=argparse.BooleanOptionalAction,
        default=yaml_cfg.get("fedsol_adaptive", False),
        help=f"Enable adaptive perturbation radius for FedSOL (default: {yaml_cfg.get('fedsol_adaptive', False)})",
    )
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=yaml_cfg.get("clip_grad_norm", None),
        help=f"Max gradient L2 norm for clipping (default: {yaml_cfg.get('clip_grad_norm', None)})",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=yaml_cfg.get("momentum", 0.0),
        help=f"Momentum factor for SGD optimizer (default: {yaml_cfg.get('momentum', 0.0)})",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=yaml_cfg.get("weight_decay", 0.0),
        help=f"Weight decay penalty (default: {yaml_cfg.get('weight_decay', 0.0)})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=yaml_cfg.get("seed", 42),
        help=f"Random seed for reproducibility (default: {yaml_cfg.get('seed', 42)})",
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=yaml_cfg.get("augment", True),
        help=f"Enable/disable training data augmentation (default: {yaml_cfg.get('augment', True)})",
    )
    parser.add_argument(
        "--random_crop",
        action=argparse.BooleanOptionalAction,
        default=yaml_cfg.get("random_crop", True),
        help=f"Enable/disable random crop during training (default: {yaml_cfg.get('random_crop', True)})",
    )
    parser.add_argument(
        "--random_flip",
        action=argparse.BooleanOptionalAction,
        default=yaml_cfg.get("random_flip", True),
        help=f"Enable/disable random horizontal flip during training (default: {yaml_cfg.get('random_flip', True)})",
    )
    parser.add_argument(
        "--no_save_metrics",
        action="store_false",
        dest="save_metrics",
        help="Disable saving local client metrics to CSV",
    )
    parser.set_defaults(save_metrics=True)
    args = parser.parse_args()

    set_seed(args.seed)

    # Resolve partition path
    if args.partition_path:
        partition_path = args.partition_path
    else:
        partition_path = str(
            get_partition_path(
                dataset_name=args.dataset,
                partition_mode=args.partition_mode,
                seed=args.seed,
                num_clients=args.num_clients,
                local_test=args.local_test,
            )
        )

    if not Path(partition_path).exists():
        print(f"ERROR: Partition file not found: {partition_path}")
        print("Run `python -m experiments.run_partition` first to generate partitions.")
        sys.exit(1)

    print(f"[Client {args.client_id}] Loading partition from {partition_path}")

    # Load dataset and partition
    train_dataset, _, test_dataset, num_classes = get_medmnist_datasets(
        args.dataset,
        download=True,
        augment=args.augment,
        random_crop=args.random_crop,
        random_flip=args.random_flip,
        load_val=False,
    )
    print(f"[Client {args.client_id}] Augmentation: augment={args.augment}, crop={args.random_crop}, flip={args.random_flip}")
    client_indices = load_partition(partition_path, split="train")

    if args.client_id >= len(client_indices):
        print(
            f"ERROR: client_id={args.client_id} but partition only has "
            f"{len(client_indices)} clients (0-{len(client_indices)-1})"
        )
        sys.exit(1)

    my_indices = client_indices[args.client_id]
    resolved_device_type = str(args.device_type).strip() if args.device_type else get_device_type(args.client_id)
    print(f"[Client {args.client_id}] Assigned {len(my_indices)} training samples")

    # Create data loaders
    train_loader = get_client_dataloader(
        train_dataset,
        my_indices,
        client_id=args.client_id,
        device_type=resolved_device_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle=True,
    )

    from torch.utils.data import DataLoader

    if args.local_test:
        client_test_indices = load_partition(partition_path, split="test")
        if client_test_indices is not None and args.client_id < len(client_test_indices):
            my_test_indices = client_test_indices[args.client_id]
            print(f"[Client {args.client_id}] Using local test partition: {len(my_test_indices)} samples")
            test_loader = get_client_dataloader(
                test_dataset,
                my_test_indices,
                client_id=args.client_id,
                device_type=resolved_device_type,
                batch_size=args.batch_size or 32,
                shuffle=False,
                num_workers=0,
            )
        else:
            print(f"[Client {args.client_id}] Warning: local_test requested but no test_indices found in partition. Falling back to global test set.")
            test_loader = DataLoader(
                test_dataset, batch_size=32, shuffle=False, num_workers=0
            )
    else:
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
        lr_mode=args.lr_mode,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        lr_min=args.lr_min,
        device_type=resolved_device_type,
        save_local_metrics=args.save_metrics,
        server_address=args.server_address,
        strategy=args.strategy,
        partition_path=partition_path,
        client_dir=args.client_dir,
        model_name=args.model,
        optimizer_name=args.optimizer,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        name=args.name,
        fedsol_rho=args.fedsol_rho,
        fedsol_target=args.fedsol_target,
        fedsol_loss_type=args.fedsol_loss_type,
        fedsol_temperature=args.fedsol_temperature,
        fedsol_adaptive=args.fedsol_adaptive,
        clip_grad_norm=args.clip_grad_norm,
    )

    print(f"[Client {args.client_id}] Connecting to {args.server_address}...")
    try:
        fl.client.start_numpy_client(
            server_address=args.server_address,
            client=client,
        )
    except KeyboardInterrupt:
        print(f"\n[Client {args.client_id}] Interrupted by user (Ctrl+C). Exiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()
