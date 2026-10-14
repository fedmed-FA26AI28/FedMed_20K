import os
import argparse
from pathlib import Path
import yaml
import numpy as np
from datasets.medmnist_code import get_medmnist_datasets
from datasets.partition import (
    dirichlet_partition, pathological_partition, practical_partition,
    save_partition, visualize_partition
)

def load_yaml_config(config_path: str = "configs/experiment.yaml") -> dict:
    """Load configuration dictionary from YAML file if available."""
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

# Benchmark grid defaults
DEFAULT_DATASETS = ["bloodmnist", "pathmnist", "organamnist"]
DEFAULT_PARTITIONS = ["dirichlet_0.3", "dirichlet_0.1", "pathological", "practical"]
DEFAULT_NUM_CLIENTS = 12
DEFAULT_SEED = 42

def run_one(dataset_name: str, partition_mode: str, num_clients: int, seed: int, local_test: bool = False):
    """Run partition for one dataset and partition mode."""
    print(f"\n{'='*65}")
    print(f"  Partitioning | dataset={dataset_name} | mode={partition_mode} | clients={num_clients} | seed={seed} | local_test={local_test}")
    print(f"{'='*65}")
    
    # Load dataset
    print(f"Loading {dataset_name} dataset...")
    train_dataset, _, test_dataset, _ = get_medmnist_datasets(dataset_name, download=True)
    print(f"Train size: {len(train_dataset)} samples | Test size: {len(test_dataset)} samples")
    
    # Run partition algorithm
    target_test_dataset = test_dataset if local_test else None
    if partition_mode.startswith("dirichlet"):
        alpha = float(partition_mode.split("_")[1])
        res = dirichlet_partition(train_dataset, num_clients, alpha, seed, test_dataset=target_test_dataset)
    elif partition_mode == "pathological":
        res = pathological_partition(train_dataset, num_clients, seed, test_dataset=target_test_dataset)
    elif partition_mode == "practical":
        res = practical_partition(train_dataset, num_clients, seed, test_dataset=target_test_dataset)
    else:
        raise ValueError(f"Unknown partition mode: {partition_mode}")

    if local_test:
        partition, test_partition = res
    else:
        partition = res
        test_partition = None
        
    # Print summary statistics
    sizes = [len(idx) for idx in partition]
    print(f"  Train samples per client: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}")
    total = sum(sizes)
    print(f"  Total train samples: {total}")
    assert total == len(train_dataset), "Total assigned train samples must equal original dataset length!"
    
    # Validate no duplicate indices across clients
    all_indices = [idx for sublist in partition for idx in sublist]
    assert len(all_indices) == len(set(all_indices)), "Found duplicate sample indices across clients in train!"

    if local_test:
        test_sizes = [len(idx) for idx in test_partition]
        print(f"  Test samples per client: min={min(test_sizes)}, max={max(test_sizes)}, mean={np.mean(test_sizes):.1f}")
        total_test = sum(test_sizes)
        print(f"  Total test samples: {total_test}")
        assert total_test == len(test_dataset), "Total assigned test samples must equal original test dataset length!"
        all_test_indices = [idx for sublist in test_partition for idx in sublist]
        assert len(all_test_indices) == len(set(all_test_indices)), "Found duplicate sample indices across clients in test!"
    print("  Validation PASSED: no duplicates, total matches.")
    
    # Save partition mapping JSON
    save_partition(
        partition,
        train_dataset,
        dataset_name,
        partition_mode,
        seed,
        num_clients,
        test_dataset=test_dataset if local_test else None,
        test_indices=test_partition if local_test else None,
        save_dir="data/partitions",
    )
    
    # Save visualization plot
    plot_tag = "_localtest" if local_test else ""
    plot_path = Path(f"data/partitions/plots/dist_{dataset_name}_{partition_mode}{plot_tag}_clients{num_clients}.png").resolve()
    try:
        visualize_partition(partition, train_dataset, dataset_name, partition_mode, num_clients, save_path=str(plot_path))
    except Exception as e:
        print(f"[Partition] Warning: Could not save partition plot: {e}")


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="configs/experiment.yaml",
                            help="Path to YAML experiment configuration file (default: configs/experiment.yaml)")
    pre_args, remaining_argv = pre_parser.parse_known_args()
    cfg = load_yaml_config(pre_args.config)

    default_dataset = cfg.get("dataset", "bloodmnist")
    default_partition = cfg.get("partition_mode", cfg.get("partition", "dirichlet_0.3"))
    default_num_clients = cfg.get("num_clients", cfg.get("min_clients", DEFAULT_NUM_CLIENTS))
    default_seed = cfg.get("seed", DEFAULT_SEED)
    default_local_test = cfg.get("local_test", False)

    parser = argparse.ArgumentParser(
        description="Run dataset partitioning.",
        parents=[pre_parser],
    )
    parser.add_argument("--dataset", type=str, default=default_dataset,
                        help=f"Dataset name (e.g., bloodmnist, pathmnist) (default: {default_dataset})")
    parser.add_argument("--partition_mode", type=str, default=default_partition,
                        help=f"Partition mode (e.g., dirichlet_0.3, pathological, practical) (default: {default_partition})")
    parser.add_argument("--num_clients", type=int, default=default_num_clients,
                        help=f"Number of clients (default: {default_num_clients})")
    parser.add_argument("--seed", type=int, default=default_seed,
                        help=f"Random seed (default: {default_seed})")
    parser.add_argument("--local_test", dest="local_test", action=argparse.BooleanOptionalAction, default=default_local_test,
                        help=f"Partition test dataset across clients (following FedSLD paper) (default: {default_local_test})")
    parser.add_argument("--all", action="store_true",
                        help="Run all defaults combinations in the matrix")
    args = parser.parse_args()

    if args.all:
        for ds in DEFAULT_DATASETS:
            for part in DEFAULT_PARTITIONS:
                try:
                    run_one(ds, part, args.num_clients, args.seed, local_test=args.local_test)
                except ValueError as e:
                    print(f"Skipping {ds} - {part} for {args.num_clients} clients: {e}")
    else:
        run_one(args.dataset, args.partition_mode, args.num_clients, args.seed, local_test=args.local_test)
        
    print("\nDone! Check data/partitions/ for JSON files and plots.")


if __name__ == "__main__":
    main()