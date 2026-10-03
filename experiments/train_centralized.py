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

from models.cnn import CNN, SimpleCNN
from datasets.medmnist_code import get_bloodmnist_datasets, get_bloodmnist_dataloaders
from client.train import train
from client.evaluate import evaluate 


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


def plot_and_save_result(train_metrics, eval_metrics, save_dir):
    """Plot and save learning curves, confusion matrix, and per-class metrics."""
    history = train_metrics['history']
    epochs = range(1, len(history['loss']) + 1)

    # 1. Loss and Accuracy Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], 'b-', marker='o', label='Train Loss')
    if 'val_loss' in history:
        plt.plot(epochs, history['val_loss'], 'r-', marker='s', label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Accuracy curve (Train vs Val)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['accuracy'], 'b-', marker='o', label='Train Accuracy')
    if 'val_accuracy' in history:
        plt.plot(epochs, history['val_accuracy'], 'r-', marker='s', label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()
    
    # 2. Confusion Matrix
    class_names = ['Basophil', 'Eosinophil', 'Erythroblast', 'IG',
                   'Lymphocyte', 'Monocyte', 'Neutrophil', 'Platelet']
    cm = np.array(eval_metrics['confusion_matrix'])
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix on Test Set')
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


def plot_summary_card(train_metrics, eval_metrics, resource_before, resource_after, dataset_info, save_dir):
    """Generate and save an executive summary card of the centralized run."""
    fig, ax = plt.subplots(figsize=(10, 12))
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
    y = draw_section_header(ax, y, 'MODEL METRICS (Test Set)', color_model)
    rows = [
        ('Loss', f"{eval_metrics['loss']:.4f}"),
        ('Accuracy', f"{eval_metrics['accuracy']:.4f}  ({eval_metrics['accuracy']*100:.2f}%)"),
        ('Precision (macro)', f"{eval_metrics['precision']:.4f}"),
        ('Recall    (macro)', f"{eval_metrics['recall']:.4f}"),
        ('F1-Score  (macro)', f"{eval_metrics['f1_score']:.4f}"),
    ]
    for i, (lbl, val) in enumerate(rows):
        y = draw_row(ax, y, lbl, val, color_row_a if i % 2 == 0 else color_row_b)

    y -= 0.01

    # Section 3: Training Parameters
    history = train_metrics['history']
    total_epochs = len(history['loss'])
    y = draw_section_header(ax, y, 'TRAINING INFO', color_train)
    rows = [
        ('Total Epochs Run', f"{total_epochs}"),
        ('Best Epoch (saved weights)', f"{train_metrics.get('best_epoch', 'N/A')}"),
        ('Best Val Loss  (best weights)', f"{train_metrics['best_val_loss']:.4f}"
                                          if isinstance(train_metrics.get('best_val_loss'), float) else 'N/A'),
        ('Best Val Acc   (best weights)', f"{train_metrics['best_val_acc']:.4f}  ({train_metrics['best_val_acc']*100:.2f}%)"
                                          if isinstance(train_metrics.get('best_val_acc'), float) else 'N/A'),
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


def run_centralized(
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    lr_patience: int = 3,
    lr_factor: float = 0.5,
    min_lr: float = 1e-6,
    early_stop_patience: int = 8,
    config_source: str = "configs/experiment.yaml",
):
    print("Starting centralized training...")
    print(
        f"Config ({config_source}): epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}, "
        f"lr_patience={lr_patience}, lr_factor={lr_factor}, early_stop_patience={early_stop_patience}"
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results/centralized/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # Load dataset
    print("Loading BloodMNIST dataset...")
    train_data, val_data, test_data, num_classes = get_bloodmnist_datasets(download=True)
    train_loader, val_loader, test_loader, num_classes = get_bloodmnist_dataloaders(batch_size=batch_size)
    print(f"Number of training samples: {len(train_data)}")
    print(f"Number of validation samples: {len(val_data)}")
    print(f"Number of test samples: {len(test_data)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CNN(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    resource_usage_before = get_resource_usage()

    print("Starting training...")
    training_metrics = train(
        model,
        train_loader,
        optimizer,
        epochs,
        device,
        val_loader=val_loader,
        lr_patience=lr_patience,
        lr_factor=lr_factor,
        min_lr=min_lr,
        early_stop_patience=early_stop_patience,
    )

    resource_usage_after = get_resource_usage()

    print("Evaluating on test set...")
    loss, eval_metrics = evaluate(model, test_loader, device)

    # Print comprehensive results summary
    print(f"\n{'='*55}")
    print(f"  FINAL RESULTS ON TEST SET")
    print(f"{'='*55}")
    print(f"  Loss            : {loss:.4f}")
    print(f"  Accuracy        : {eval_metrics['accuracy']:.4f}")
    print(f"  Precision(macro): {eval_metrics['precision']:.4f}")
    print(f"  Recall   (macro): {eval_metrics['recall']:.4f}")
    print(f"  F1-Score (macro): {eval_metrics['f1_score']:.4f}")
    print(f"  Train Time      : {training_metrics['training_time']:.1f}s")
    print(f"{'='*55}")

    plot_and_save_result(training_metrics, eval_metrics, save_dir)

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
        save_dir=save_dir
    )

    results = {
        "config_source": config_source,
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "lr_patience": lr_patience,
            "lr_factor": lr_factor,
            "min_lr": min_lr,
            "early_stop_patience": early_stop_patience,
        },
        "dataset_info": {
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data),
            "num_classes": num_classes
        },
        "model_metrics": eval_metrics,
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

    default_epochs = cen_cfg.get("epochs", yaml_cfg.get("centralized_epochs", 100))
    default_batch_size = cen_cfg.get("batch_size", yaml_cfg.get("centralized_batch_size", 32))
    default_lr = cen_cfg.get("learning_rate", yaml_cfg.get("centralized_learning_rate", yaml_cfg.get("learning_rate", 0.001)))
    default_lr_patience = cen_cfg.get("lr_patience", yaml_cfg.get("centralized_lr_patience", 3))
    default_lr_factor = cen_cfg.get("lr_factor", yaml_cfg.get("centralized_lr_factor", 0.5))
    default_min_lr = cen_cfg.get("min_lr", yaml_cfg.get("centralized_min_lr", 1e-6))
    default_early_stop = cen_cfg.get("early_stop_patience", yaml_cfg.get("centralized_early_stop_patience", 8))

    parser = argparse.ArgumentParser(
        description="Centralized Training Baseline for BloodMNIST with YAML Config Support.",
        parents=[pre_parser],
    )
    parser.add_argument("--epochs", type=int, default=default_epochs,
                        help=f"Maximum training epochs (default: {default_epochs}).")
    parser.add_argument("--batch_size", type=int, default=default_batch_size,
                        help=f"Batch size (default: {default_batch_size}).")
    parser.add_argument("--lr", "--learning_rate", type=float, default=default_lr, dest="learning_rate",
                        help=f"Initial learning rate for Adam optimizer (default: {default_lr}).")
    parser.add_argument("--lr_patience", type=int, default=default_lr_patience,
                        help=f"Epochs without val loss improvement before reducing LR (default: {default_lr_patience}).")
    parser.add_argument("--lr_factor", type=float, default=default_lr_factor,
                        help=f"Multiplicative factor for LR reduction (default: {default_lr_factor}).")
    parser.add_argument("--min_lr", type=float, default=default_min_lr,
                        help=f"Minimum learning rate lower bound (default: {default_min_lr}).")
    parser.add_argument("--early_stop_patience", type=int, default=default_early_stop,
                        help=f"Early stopping patience in epochs (default: {default_early_stop}).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_centralized(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        min_lr=args.min_lr,
        early_stop_patience=args.early_stop_patience,
        config_source=args.config,
    )