"""Script to validate and execute partitions across different datasets and partition modes."""

import argparse
import numpy as np
from datasets.medmnist_code import get_medmnist_datasets
from datasets.partition import (
    dirichlet_partition, pathological_partition, practical_partition,
    save_partition, visualize_partition
)

# Benchmark grid defaults
DEFAULT_DATASETS = ["bloodmnist", "pathmnist", "organamnist"]
DEFAULT_PARTITIONS = ["dirichlet_0.3", "dirichlet_0.1", "pathological", "practical"]
DEFAULT_NUM_CLIENTS = 12
DEFAULT_SEED = 42

def run_one(dataset_name: str, partition_mode: str, num_clients: int, seed: int):
    """Run partition for one dataset and partition mode."""
    print(f"\n{'='*65}")
    print(f"  Partitioning | dataset={dataset_name} | mode={partition_mode} | clients={num_clients} | seed={seed}")
    print(f"{'='*65}")
    
    # Load dataset
    print(f"Loading {dataset_name} train dataset...")
    train_dataset, _, _, _ = get_medmnist_datasets(dataset_name, download=True)
    print(f"Train size: {len(train_dataset)} samples")
    
    # Run partition algorithm
    if partition_mode.startswith("dirichlet"):
        alpha = float(partition_mode.split("_")[1])
        partition = dirichlet_partition(train_dataset, num_clients, alpha, seed)
    elif partition_mode == "pathological":
        partition = pathological_partition(train_dataset, num_clients, seed)
    elif partition_mode == "practical":
        partition = practical_partition(train_dataset, num_clients, seed)
    else:
        raise ValueError(f"Unknown partition mode: {partition_mode}")
        
    # Print summary statistics
    sizes = [len(idx) for idx in partition]
    print(f"  Samples per client: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}")
    total = sum(sizes)
    print(f"  Total samples: {total}")
    assert total == len(train_dataset), "Total assigned samples must equal original dataset length!"
    
    # Validate no duplicate indices across clients
    all_indices = [idx for sublist in partition for idx in sublist]
    assert len(all_indices) == len(set(all_indices)), "Found duplicate sample indices across clients!"
    print("  Validation PASSED: no duplicates, total matches.")
    
    # Save partition mapping JSON
    save_partition(partition, train_dataset, dataset_name, partition_mode, seed, num_clients,
                   save_dir="data/partitions")
    
    # Save visualization plot
    plot_path = f"data/partitions/plots/dist_{dataset_name}_{partition_mode}_clients{num_clients}.png"
    visualize_partition(partition, train_dataset, dataset_name, partition_mode, num_clients, save_path=plot_path)


def main():
    parser = argparse.ArgumentParser(description="Run dataset partitioning.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (e.g., bloodmnist, pathmnist)")
    parser.add_argument("--partition_mode", type=str, default=None,
                        help="Partition mode (e.g., dirichlet_0.3, pathological)")
    parser.add_argument("--num_clients", type=int, default=DEFAULT_NUM_CLIENTS,
                        help="Number of clients (default: 12)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed (default: 42)")
    parser.add_argument("--all", action="store_true",
                        help="Run all defaults combinations in the matrix")
    args = parser.parse_args()

    if args.all or (args.dataset is None and args.partition_mode is None):
        for ds in DEFAULT_DATASETS:
            for part in DEFAULT_PARTITIONS:
                # practical mode strictly requires 12 clients, skip if testing num_clients != 12
                # But our default is 12, so we are safe
                try:
                    run_one(ds, part, args.num_clients, args.seed)
                except ValueError as e:
                    print(f"Skipping {ds} - {part} for {args.num_clients} clients: {e}")
    else:
        ds = args.dataset if args.dataset else "bloodmnist"
        part = args.partition_mode if args.partition_mode else "dirichlet_0.3"
        run_one(ds, part, args.num_clients, args.seed)
        
    print("\nDone! Check data/partitions/ for JSON files and plots.")


if __name__ == "__main__":
    main()