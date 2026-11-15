# Centralized Baseline Training Pipeline for BloodMNIST

import os
import sys
import json
import argparse
import datetime

# Ensure Windows consoles don't crash on character encoding
if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import psutil
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from models import get_model, set_seed, count_parameters
from datasets.medmnist_code import get_medmnist_datasets, get_medmnist_dataloaders
from client.train import train
from client.evaluate import evaluate 
from utils.path_utils import sanitize_run_name 


def get_resource_usage():
    """Retrieve current hardware resource usage (RAM, CPU, GPU)."""
    ram_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent()
    gpu_memory_allocated = 0
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
    return {
        "ram_usage": ram_usage,
        "cpu_usage": cpu_usage,
        "gpu_memory_allocated_MB": gpu_memory_allocated
    }


def plot_and_save_result(train_metrics, eval_metrics, save_dir, dataset_name="bloodmnist"):
    """Plot and save learning curves, confusion matrix, and per-class metrics."""
    history = train_metrics['history']
    epochs = range(1, len(history['loss']) + 1)

    # 1. Loss and Accuracy Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], 'b-', marker='o', markersize=3, label='Train Loss')
    if 'val_loss' in history:
        plt.plot(epochs, history['val_loss'], 'r-', marker='s', markersize=3, label='Val Loss')
    if 'test_loss' in history:
        plt.plot(epochs, history['test_loss'], 'g-', marker='^', markersize=3, label='Test Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Accuracy curve (Train vs Val vs Test / BTA)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['accuracy'], 'b-', marker='o', markersize=3, label='Train Accuracy')
    if 'val_accuracy' in history:
        plt.plot(epochs, history['val_accuracy'], 'r-', marker='s', markersize=3, label='Val Accuracy')
    if 'test_accuracy' in history:
        plt.plot(epochs, history['test_accuracy'], 'g-', marker='^', markersize=3, label='Test Accuracy')
        cum_bta = []
        best_so_far = 0.0
        for acc in history['test_accuracy']:
            best_so_far = max(best_so_far, acc)
            cum_bta.append(best_so_far)
        plt.plot(epochs, cum_bta, color='#7c3aed', linestyle='--', linewidth=1.8, label='Best Test Accuracy (BTA)')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()
    
    # 2. Confusion Matrix
    cm = np.array(eval_metrics['confusion_matrix'])
    num_cm_classes = cm.shape[0]
    bloodmnist_names = ['Basophil', 'Eosinophil', 'Erythroblast', 'IG',
                        'Lymphocyte', 'Monocyte', 'Neutrophil', 'Platelet']
    if num_cm_classes == 8 and str(dataset_name).lower().strip() == "bloodmnist":
        class_names = bloodmnist_names
    else:
        class_names = [f"Class {i}" for i in range(num_cm_classes)]
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix on Test Set (Best Model)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    
    # 3. Precision, Recall, F1-Score per Class
    precision_class = eval_metrics['precision_class']
    recall_class = eval_metrics['recall_class']
    f1_score_class = eval_metrics['f1_score_class']
    
    num_classes = len(f1_score_class)
    classes = np.arange(num_classes)
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    plt.bar(classes - width, precision_class, width, label='Precision', color='lightcoral')
    plt.bar(classes, recall_class, width, label='Recall', color='lightgreen')
    plt.bar(classes + width, f1_score_class, width, label='F1-Score', color='skyblue')
    
    plt.title('Precision, Recall, and F1-Score per Class')
    plt.xlabel('Class ID')
    plt.ylabel('Score')
    plt.xticks(classes)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_class_metrics.png'))
    plt.close()


def plot_summary_card(
    train_metrics,
    eval_metrics,
    resource_before,
    resource_after,
    dataset_info,
    save_dir,
    model_info=None,
    final_eval_metrics=None,
):
    """Generate and save an executive summary card of the centralized run."""
    fig, ax = plt.subplots(figsize=(10, 13))
    ax.axis('off')

    # Color scheme
    color_title = '#2C3E50'
    color_model = '#1A5276'
    color_train = '#145A32'
    color_resource = '#6E2F7E'
    color_data = '#7D6608'
    color_row_a = '#EBF5FB'
    color_row_b = '#FDFEFE'

    y = 0.98

    def draw_section_header(ax, y, title, color):
        ax.add_patch(plt.Rectangle((0, y - 0.025), 1, 0.03,
                                   transform=ax.transAxes, color=color, zorder=2))
        ax.text(0.5, y - 0.01, title, transform=ax.transAxes,
                ha='center', va='center', fontsize=12, fontweight='bold',
                color='white', zorder=3)
        return y - 0.025

    def draw_row(ax, y, label, value, bg_color):
        ax.add_patch(plt.Rectangle((0, y - 0.025), 1, 0.026,
                                   transform=ax.transAxes, color=bg_color, zorder=1))
        ax.text(0.05, y - 0.012, label, transform=ax.transAxes,
                ha='left', va='center', fontsize=10, color='#2C3E50')
        ax.text(0.95, y - 0.012, value, transform=ax.transAxes,
                ha='right', va='center', fontsize=10, fontweight='bold', color='#1A1A1A')
        return y - 0.026

    # Main Title
    ax.add_patch(plt.Rectangle((0, y - 0.04), 1, 0.045,
                               transform=ax.transAxes, color=color_title, zorder=2))
    ax.text(0.5, y - 0.018, 'CENTRALIZED BASELINE - RESULT SUMMARY',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=13, fontweight='bold', color='white', zorder=3)
    y -= 0.05

    # Section 1: Dataset Info
    y = draw_section_header(ax, y, 'DATASET INFO', color_data)
    rows = [
        ('Train Samples', f"{dataset_info['train_samples']:,}"),
        ('Val Samples', f"{dataset_info['val_samples']:,}"),
        ('Test Samples', f"{dataset_info['test_samples']:,}"),
        ('Num Classes', f"{dataset_info['num_classes']}"),
    ]
    for i, (lbl, val) in enumerate(rows):
        y = draw_row(ax, y, lbl, val, color_row_a if i % 2 == 0 else color_row_b)

    y -= 0.01

    # Section 2: Model Performance Metrics
    y = draw_section_header(ax, y, 'MODEL METRICS & SPECS (Test Set)', color_model)
    rows = []
    if model_info:
        m_name = str(model_info.get("model_name", "CNN")).upper()
        t_params = model_info.get("total_parameters")
        p_str = f"{t_params:,} params" if t_params is not None else "N/A"
        w_kb = model_info.get("weight_size_kb")
        w_str = f"{w_kb:.2f} KB ({w_kb/1024.0:.3f} MB)" if w_kb is not None else "N/A"
        rows.extend([
            ('Model Architecture', m_name),
            ('Total Parameters', p_str),
            ('Model Weight Size', w_str),
        ])

    bta = train_metrics.get('bta', eval_metrics['accuracy'])
    bta_ep = train_metrics.get('bta_epoch', train_metrics.get('best_epoch'))
    bta_loss = train_metrics.get('bta_loss', eval_metrics['loss'])
    ta = (final_eval_metrics['accuracy'] if final_eval_metrics is not None
          else train_metrics.get('final_test_accuracy', eval_metrics['accuracy']))
    ta_loss = (final_eval_metrics['loss'] if final_eval_metrics is not None
               else train_metrics.get('final_test_loss', eval_metrics['loss']))
    bta_suffix = f"  @ E{bta_ep}" if bta_ep is not None else ""

    rows.extend([
        ('Best Test Acc (BTA)', f"{bta:.4f}  ({bta*100:.2f}%){bta_suffix}"),
        ('Best Test Loss (at BTA)', f"{bta_loss:.4f}" if bta_loss is not None else "N/A"),
        ('Final Test Acc (TA)', f"{ta:.4f}  ({ta*100:.2f}%)"),
        ('Final Test Loss', f"{ta_loss:.4f}" if ta_loss is not None else "N/A"),
        ('Precision (macro)', f"{eval_metrics['precision']:.4f}"),
        ('Recall    (macro)', f"{eval_metrics['recall']:.4f}"),
        ('F1-Score  (macro)', f"{eval_metrics['f1_score']:.4f}"),
    ])
    for i, (lbl, val) in enumerate(rows):
        y = draw_row(ax, y, lbl, val, color_row_a if i % 2 == 0 else color_row_b)

    y -= 0.01

    # Section 3: Training Parameters
    history = train_metrics['history']
    total_epochs = len(history['loss'])
    best_val_loss = train_metrics.get('best_val_loss')
    best_val_acc = train_metrics.get('best_val_acc')
    y = draw_section_header(ax, y, 'TRAINING INFO', color_train)
    rows = [
        ('Total Epochs Run', f"{total_epochs}"),
        ('Best Epoch (saved weights)', f"{train_metrics.get('best_epoch', 'N/A')}"),
        ('Best Val Loss  (best val)', f"{best_val_loss:.4f}"
                                      if isinstance(best_val_loss, (int, float)) and best_val_loss < float('inf') else 'N/A'),
        ('Best Val Acc   (best val)', f"{best_val_acc:.4f}  ({best_val_acc*100:.2f}%)"
                                      if isinstance(best_val_acc, (int, float)) and best_val_acc >= 0.0 else 'N/A'),
        ('Final Train Loss (last epoch)', f"{train_metrics['final_loss']:.4f}"),
        ('Training Time', f"{train_metrics['training_time']:.1f} s  ({train_metrics['training_time']/60:.1f} min)"),
    ]
    for i, (lbl, val) in enumerate(rows):
        y = draw_row(ax, y, lbl, val, color_row_a if i % 2 == 0 else color_row_b)

    y -= 0.01

    # Section 4: Hardware Resources
    y = draw_section_header(ax, y, 'HARDWARE RESOURCES', color_resource)
    rows = [
        ('CPU Usage (before)', f"{resource_before['cpu_usage']:.1f}%"),
        ('CPU Usage (after)', f"{resource_after['cpu_usage']:.1f}%"),
        ('RAM Usage (before)', f"{resource_before['ram_usage']:.1f}%"),
        ('RAM Usage (after)', f"{resource_after['ram_usage']:.1f}%"),
        ('GPU Memory (before)', f"{resource_before['gpu_memory_allocated_MB']:.1f} MB"),
        ('GPU Memory (after)', f"{resource_after['gpu_memory_allocated_MB']:.1f} MB"),
    ]
    for i, (lbl, val) in enumerate(rows):
        y = draw_row(ax, y, lbl, val, color_row_a if i % 2 == 0 else color_row_b)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'summary_card.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Summary card saved: summary_card.png")


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


def _save_state_dict(state_dict, filepath):
    """Persist a model state_dict on CPU."""
    cpu_state = {k: v.cpu().clone() for k, v in state_dict.items()}
    torch.save(cpu_state, filepath)


def run_centralized(
    dataset_name: str = "bloodmnist",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    lr_mode: str = "fixed",
    lr_patience: int = 3,
    lr_factor: float = 0.5,
    min_lr: float = 1e-6,
    lr_decay_steps: int = 15,
    lr_decay_gamma: float = 0.5,
    early_stop_patience: int = 8,
    early_stop_metric: str = "accuracy",
    config_source: str = "configs/experiment.yaml",
    model_name: str = "cnn",
    seed: int = 42,
    optimizer_name: str = "adam",
    momentum: float = 0.0,
    weight_decay: float = 1e-4,
    clip_grad_norm: float = None,
    augment: bool = True,
    random_crop: bool = True,
    random_flip: bool = True,
    name: str = None,
    save_dir: str = None,
):
    set_seed(seed)
    opt_name = str(optimizer_name).lower().strip()
    wd = float(weight_decay) if weight_decay is not None else 0.0
    clip_val = float(clip_grad_norm) if clip_grad_norm is not None else None
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = sanitize_run_name(name) if name else None

    if save_dir:
        final_save_dir = str(save_dir)
    elif run_name:
        final_save_dir = f"results/centralized/{run_name}"
    else:
        final_save_dir = f"results/centralized/{timestamp}"
    os.makedirs(final_save_dir, exist_ok=True)
    save_dir = final_save_dir

    print("Starting centralized training...")
    print(
        f"Config ({config_source}): dataset={dataset_name}, model={model_name}, optimizer={opt_name.upper()}"
        + (f"(momentum={momentum})" if opt_name == "sgd" else "")
        + f", weight_decay={wd}, clip_grad_norm={clip_val}, run_name={run_name or timestamp}, seed={seed}, "
        + f"epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}, lr_mode={lr_mode}, "
        + f"lr_patience={lr_patience}, lr_factor={lr_factor}, lr_decay_steps={lr_decay_steps}, lr_decay_gamma={lr_decay_gamma}, "
        + f"early_stop_patience={early_stop_patience}, early_stop_metric={early_stop_metric}, "
        + f"augment={augment}, random_crop={random_crop}, random_flip={random_flip}"
    )

    # Load dataset
    print(f"Loading {dataset_name} dataset...")
    train_data, val_data, test_data, num_classes = get_medmnist_datasets(
        dataset_name=dataset_name,
        download=True,
        augment=augment,
        random_crop=random_crop,
        random_flip=random_flip,
    )
    train_loader, val_loader, test_loader, num_classes = get_medmnist_dataloaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        augment=augment,
        random_crop=random_crop,
        random_flip=random_flip,
    )
    print(f"Number of training samples: {len(train_data)}")
    print(f"Number of validation samples: {len(val_data)}")
    print(f"Number of test samples: {len(test_data)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_model(model_name, num_classes=num_classes).to(device)
    if opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=wd)
    elif opt_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=wd)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)

    resource_usage_before = get_resource_usage()

    print("Starting training...")
    training_metrics = train(
        model,
        train_loader,
        optimizer,
        epochs,
        device,
        val_loader=val_loader,
        test_loader=test_loader,
        lr_mode=lr_mode,
        lr_patience=lr_patience,
        lr_factor=lr_factor,
        min_lr=min_lr,
        lr_decay_steps=lr_decay_steps,
        lr_decay_gamma=lr_decay_gamma,
        early_stop_patience=early_stop_patience,
        early_stop_metric=early_stop_metric,
        clip_grad_norm=clip_val,
    )

    resource_usage_after = get_resource_usage()

    best_weights = training_metrics.get("best_weights")
    final_weights = training_metrics.get("final_weights")
    total_epochs_run = len(training_metrics["history"]["loss"])

    print("Evaluating best model on test set...")
    best_loss, eval_metrics = evaluate(model, test_loader, device)

    if (
        best_weights is not None
        and final_weights is not None
        and training_metrics.get("best_epoch") != total_epochs_run
    ):
        model.load_state_dict(final_weights)
        final_loss, final_eval_metrics = evaluate(model, test_loader, device)
        model.load_state_dict(best_weights)
    else:
        final_loss, final_eval_metrics = best_loss, dict(eval_metrics)

    best_ckpt_state = best_weights if best_weights is not None else model.state_dict()
    final_ckpt_state = final_weights if final_weights is not None else model.state_dict()
    _save_state_dict(best_ckpt_state, os.path.join(save_dir, "best_model.pth"))
    _save_state_dict(final_ckpt_state, os.path.join(save_dir, "final_model.pth"))

    bta_val = float(training_metrics.get("bta", eval_metrics["accuracy"]))
    bta_epoch_val = training_metrics.get("bta_epoch", training_metrics.get("best_epoch"))
    bta_loss_val = float(training_metrics.get("bta_loss", best_loss))
    ta_val = float(final_eval_metrics["accuracy"])
    final_test_loss_val = float(final_loss)

    eval_metrics["bta"] = bta_val
    eval_metrics["bta_epoch"] = bta_epoch_val
    eval_metrics["bta_loss"] = bta_loss_val
    eval_metrics["ta"] = ta_val
    eval_metrics["final_test_loss"] = final_test_loss_val

    # Print comprehensive results summary
    print(f"\n{'='*55}")
    print(f"  FINAL RESULTS ON TEST SET")
    print(f"{'='*55}")
    print(f"  Best Test Acc (BTA) : {bta_val:.4f} ({bta_val * 100:.2f}%) @ Epoch {bta_epoch_val}")
    print(f"  Best Test Loss      : {bta_loss_val:.4f}")
    print(f"  Final Test Acc (TA) : {ta_val:.4f} ({ta_val * 100:.2f}%)")
    print(f"  Final Test Loss     : {final_test_loss_val:.4f}")
    print(f"  Precision(macro)    : {eval_metrics['precision']:.4f}")
    print(f"  Recall   (macro)    : {eval_metrics['recall']:.4f}")
    print(f"  F1-Score (macro)    : {eval_metrics['f1_score']:.4f}")
    print(f"  Train Time          : {training_metrics['training_time']:.1f}s")
    print(f"{'='*55}")

    plot_and_save_result(training_metrics, eval_metrics, save_dir, dataset_name=dataset_name)

    tot_params, _ = count_parameters(model)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_info_card = {
        "model_name": model_name,
        "total_parameters": tot_params,
        "weight_size_kb": param_bytes / 1024.0,
    }

    plot_summary_card(
        train_metrics=training_metrics,
        eval_metrics=eval_metrics,
        resource_before=resource_usage_before,
        resource_after=resource_usage_after,
        dataset_info={
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data),
            "num_classes": num_classes
        },
        save_dir=save_dir,
        model_info=model_info_card,
        final_eval_metrics=final_eval_metrics,
    )

    results = {
        "run_name": run_name or timestamp,
        "config_source": config_source,
        "model": model_name,
        "optimizer": opt_name,
        "momentum": momentum,
        "weight_decay": wd,
        "seed": seed,
        "hyperparameters": {
            "model": model_name,
            "optimizer": opt_name,
            "momentum": momentum,
            "weight_decay": wd,
            "clip_grad_norm": clip_val,
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "lr_mode": lr_mode,
            "lr_patience": lr_patience,
            "lr_factor": lr_factor,
            "min_lr": min_lr,
            "lr_decay_steps": lr_decay_steps,
            "lr_decay_gamma": lr_decay_gamma,
            "early_stop_patience": early_stop_patience,
            "early_stop_metric": early_stop_metric,
            "augment": augment,
            "random_crop": random_crop,
            "random_flip": random_flip,
        },
        "dataset_info": {
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data),
            "num_classes": num_classes
        },
        "model_metrics": eval_metrics,
        "training_summary": {
            "total_epochs_run": total_epochs_run,
            "best_epoch": training_metrics.get("best_epoch"),
            "best_test_accuracy": bta_val,
            "bta": bta_val,
            "bta_epoch": bta_epoch_val,
            "bta_loss": bta_loss_val,
            "best_test_loss": training_metrics.get("best_test_loss"),
            "final_test_accuracy": ta_val,
            "ta": ta_val,
            "final_test_loss": final_test_loss_val,
            "best_val_acc": training_metrics.get("best_val_acc"),
            "best_val_loss": training_metrics.get("best_val_loss"),
            "final_train_loss": training_metrics.get("final_loss"),
            "final_train_accuracy": training_metrics.get("final_accuracy"),
            "history": training_metrics.get("history", {}),
        },
        "resource_metrics": {
            "training_time_seconds": training_metrics["training_time"],
            "hardware_before_train": resource_usage_before,
            "hardware_after_train": resource_usage_after
        }
    }

    with open(os.path.join(save_dir, "centralized_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Results saved to: {save_dir}")


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment.yaml",
        help="Path to YAML experiment configuration file (default: configs/experiment.yaml)",
    )
    pre_args, remaining_argv = pre_parser.parse_known_args()
    yaml_cfg = load_yaml_config(pre_args.config)

    # Resolve centralized run parameters from YAML (nested under 'centralized' or flat 'centralized_*')
    cen_cfg = yaml_cfg.get("centralized", {})
    if not isinstance(cen_cfg, dict):
        cen_cfg = {}

    default_dataset = cen_cfg.get("dataset", yaml_cfg.get("dataset", "bloodmnist"))
    default_epochs = cen_cfg.get("epochs", yaml_cfg.get("centralized_epochs", 100))
    default_batch_size = cen_cfg.get("batch_size", yaml_cfg.get("centralized_batch_size", 32))
    default_lr = cen_cfg.get("learning_rate", yaml_cfg.get("centralized_learning_rate", yaml_cfg.get("learning_rate", 0.001)))
    default_lr_mode = cen_cfg.get("lr_mode", yaml_cfg.get("centralized_lr_mode", yaml_cfg.get("lr_mode", "fixed")))
    default_lr_patience = cen_cfg.get("lr_patience", yaml_cfg.get("centralized_lr_patience", 3))
    default_lr_factor = cen_cfg.get("lr_factor", yaml_cfg.get("centralized_lr_factor", 0.5))
    default_min_lr = cen_cfg.get("min_lr", yaml_cfg.get("centralized_min_lr", 1e-6))
    default_lr_decay_steps = cen_cfg.get("lr_decay_steps", yaml_cfg.get("centralized_lr_decay_steps", yaml_cfg.get("lr_decay_steps", 15)))
    default_lr_decay_gamma = cen_cfg.get("lr_decay_gamma", yaml_cfg.get("centralized_lr_decay_gamma", yaml_cfg.get("lr_decay_gamma", 0.5)))
    default_early_stop = cen_cfg.get("early_stop_patience", yaml_cfg.get("centralized_early_stop_patience", 8))
    default_early_stop_metric = cen_cfg.get("early_stop_metric", yaml_cfg.get("centralized_early_stop_metric", yaml_cfg.get("early_stop_metric", "accuracy")))
    default_model = cen_cfg.get("model", yaml_cfg.get("model", "cnn"))
    default_optimizer = cen_cfg.get("optimizer", yaml_cfg.get("optimizer", "adam"))
    default_momentum = cen_cfg.get("momentum", yaml_cfg.get("momentum", 0.0))
    default_weight_decay = cen_cfg.get("weight_decay", yaml_cfg.get("centralized_weight_decay", yaml_cfg.get("weight_decay", 1e-4)))
    default_clip_grad_norm = cen_cfg.get("clip_grad_norm", yaml_cfg.get("centralized_clip_grad_norm", yaml_cfg.get("clip_grad_norm", None)))
    default_seed = cen_cfg.get("seed", yaml_cfg.get("seed", 42))
    default_name = cen_cfg.get("name", yaml_cfg.get("name", None))
    default_augment = cen_cfg.get("augment", yaml_cfg.get("augment", True))
    default_crop = cen_cfg.get("random_crop", yaml_cfg.get("random_crop", True))
    default_flip = cen_cfg.get("random_flip", yaml_cfg.get("random_flip", True))

    parser = argparse.ArgumentParser(
        description="Centralized Training Baseline with YAML Config Support.",
        parents=[pre_parser],
    )
    parser.add_argument("--name", "-n", type=str, default=default_name,
                        help=f"Designated run name for output directory (default: {default_name or 'timestamp'}).")
    parser.add_argument("--output_dir", "--save_dir", dest="save_dir", type=str, default=None,
                        help="Explicit directory path to save centralized artifacts (overrides default results/centralized/<name>).")
    parser.add_argument("--dataset", type=str, default=default_dataset,
                        help=f"Dataset name (e.g. 'bloodmnist', 'pathmnist', 'organamnist', 'dermamnist') (default: {default_dataset}).")
    parser.add_argument("--model", type=str, default=default_model, choices=["cnn", "baseline", "paper", "classic_cnn", "baseline_bn", "baselinecnn_bn", "classic_bn"],
                        help=f"Model architecture (default: {default_model}).")
    parser.add_argument("--optimizer", type=str, default=default_optimizer, choices=["adam", "sgd", "adamw"],
                        help=f"Optimizer type: 'adam', 'sgd', or 'adamw' (default: {default_optimizer}).")
    parser.add_argument("--momentum", type=float, default=default_momentum,
                        help=f"Momentum for SGD (default: {default_momentum}).")
    parser.add_argument("--weight_decay", type=float, default=default_weight_decay,
                        help=f"L2 weight decay regularization coefficient (default: {default_weight_decay}).")
    parser.add_argument("--clip_grad_norm", type=float, default=default_clip_grad_norm,
                        help=f"Max L2 gradient norm for clipping (None or <= 0 disables) (default: {default_clip_grad_norm}).")
    parser.add_argument("--seed", type=int, default=default_seed,
                        help=f"Random seed for reproducibility (default: {default_seed}).")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=default_augment,
                        help=f"Enable/disable training data augmentation (default: {default_augment}).")
    parser.add_argument("--random_crop", action=argparse.BooleanOptionalAction, default=default_crop,
                        help=f"Enable/disable random crop during training (default: {default_crop}).")
    parser.add_argument("--random_flip", action=argparse.BooleanOptionalAction, default=default_flip,
                        help=f"Enable/disable random horizontal flip during training (default: {default_flip}).")
    parser.add_argument("--epochs", type=int, default=default_epochs,
                        help=f"Maximum training epochs (default: {default_epochs}).")
    parser.add_argument("--batch_size", type=int, default=default_batch_size,
                        help=f"Batch size (default: {default_batch_size}).")
    parser.add_argument("--lr", "--learning_rate", type=float, default=default_lr, dest="learning_rate",
                        help=f"Initial learning rate for optimizer (default: {default_lr}).")
    parser.add_argument("--lr_mode", type=str, default=default_lr_mode,
                        choices=["fixed", "client_loss", "val_loss", "plateau", "server_decay"],
                        help=f"LR scheduling mode (default: {default_lr_mode}).")
    parser.add_argument("--lr_patience", type=int, default=default_lr_patience,
                        help=f"Epochs without val loss improvement before reducing LR (default: {default_lr_patience}).")
    parser.add_argument("--lr_factor", type=float, default=default_lr_factor,
                        help=f"Multiplicative factor for LR reduction (default: {default_lr_factor}).")
    parser.add_argument("--min_lr", type=float, default=default_min_lr,
                        help=f"Minimum learning rate lower bound (default: {default_min_lr}).")
    parser.add_argument("--lr_decay_steps", type=int, default=default_lr_decay_steps,
                        help=f"Epoch interval for server_decay step reduction (default: {default_lr_decay_steps}).")
    parser.add_argument("--lr_decay_gamma", type=float, default=default_lr_decay_gamma,
                        help=f"Multiplicative decay factor for server_decay mode (default: {default_lr_decay_gamma}).")
    parser.add_argument("--early_stop_patience", type=int, default=default_early_stop,
                        help=f"Early stopping patience in epochs (0 = disabled) (default: {default_early_stop}).")
    parser.add_argument("--early_stop_metric", type=str, default=default_early_stop_metric,
                        choices=["accuracy", "loss"],
                        help=f"Metric monitored for best checkpoint and early stopping (default: {default_early_stop_metric}).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_centralized(
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_mode=args.lr_mode,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        min_lr=args.min_lr,
        lr_decay_steps=args.lr_decay_steps,
        lr_decay_gamma=args.lr_decay_gamma,
        early_stop_patience=args.early_stop_patience,
        early_stop_metric=args.early_stop_metric,
        config_source=args.config,
        model_name=args.model,
        optimizer_name=args.optimizer,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        clip_grad_norm=args.clip_grad_norm,
        seed=args.seed,
        augment=args.augment,
        random_crop=args.random_crop,
        random_flip=args.random_flip,
        name=args.name,
        save_dir=args.save_dir,
    )