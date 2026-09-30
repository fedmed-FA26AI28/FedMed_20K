"""Script to validate and execute Dirichlet partition across num_clients x alpha combinations."""

import argparse
import numpy as np
from datasets.medmnist_code import get_bloodmnist_datasets
from datasets.partition import (
    dirichlet_partition, save_partition, visualize_partition
)

# Benchmark grid defaults
DEFAULT_NUM_CLIENTS = [3, 5, 10]
DEFAULT_ALPHAS = [1.0, 0.3, 0.1]
DEFAULT_SEED = 42


def run_one(train_dataset, num_clients: int, alpha: float, seed: int):
    """Run partition for one (num_clients, alpha) combination."""
    print(f"\n{'='*55}")
    print(f"  Partitioning | num_clients={num_clients} | alpha={alpha} | seed={seed}")
    print(f"{'='*55}")
    partition = dirichlet_partition(train_dataset, num_clients, alpha, seed)
    
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
    save_partition(partition, train_dataset, alpha, seed, num_clients,
                   save_dir="data/partitions")
    
    # Save visualization plot
    plot_path = f"data/partitions/plots/dist_alpha{alpha}_clients{num_clients}.png"
    visualize_partition(partition, train_dataset, alpha, num_clients, save_path=plot_path)


def main():
    parser = argparse.ArgumentParser(description="Run Dirichlet non-IID partition.")
    parser.add_argument("--num_clients", type=int, default=None,
                        help="Number of clients (default: run [3, 5, 10])")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Dirichlet concentration alpha (default: run [1.0, 0.3, 0.1])")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed (default: 42)")
    parser.add_argument("--all", action="store_true",
                        help="Run all 9 combinations in the matrix")
    args = parser.parse_args()

    print("Loading BloodMNIST train dataset...")
    train_dataset, _, _, _ = get_bloodmnist_datasets(download=True)
    print(f"Train size: {len(train_dataset)} samples")

    if args.all or (args.num_clients is None and args.alpha is None):
        for nc in DEFAULT_NUM_CLIENTS:
            for al in DEFAULT_ALPHAS:
                run_one(train_dataset, nc, al, args.seed)
    else:
        nc = args.num_clients if args.num_clients else 10
        al = args.alpha if args.alpha else 0.3
        run_one(train_dataset, nc, al, args.seed)
    print("\nDone! Check data/partitions/ for JSON files and plots.")


if __name__ == "__main__":
    main()