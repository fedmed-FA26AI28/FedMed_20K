"""Preflight the study's train-only client partitions without loading images."""

from __future__ import annotations

import argparse
import json

import numpy as np

from datasets.partition import dirichlet_partition, practical_partition
from research.data import archive_path


class LabelsOnly:
    def __init__(self, labels: np.ndarray):
        self.labels = labels.reshape(-1, 1)

    def __len__(self):
        return len(self.labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["bloodmnist", "pathmnist", "organamnist"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--clients", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()
    for flag in args.datasets:
        with np.load(archive_path(flag)) as archive:
            dataset = LabelsOnly(archive["train_labels"].reshape(-1))
        labels = dataset.labels.reshape(-1)
        classes = np.unique(labels)
        for partition in ("practical", "dirichlet"):
            for seed in args.seeds:
                groups = (practical_partition(dataset, args.clients, seed=seed)
                          if partition == "practical" else
                          dirichlet_partition(dataset, args.clients, args.alpha, seed=seed))
                indices = [index for group in groups for index in group]
                if len(indices) != len(dataset) or len(set(indices)) != len(dataset):
                    raise ValueError(f"{flag} {partition} seed {seed}: duplicate or missing train rows")
                if any(len(group) == 0 for group in groups):
                    raise ValueError(f"{flag} {partition} seed {seed}: empty client")
                counts = np.array([np.bincount(labels[group], minlength=len(classes))
                                   for group in groups])
                print(json.dumps({"dataset": flag, "partition": partition, "seed": seed,
                                  "train_rows": len(dataset),
                                  "smallest_client": min(map(len, groups)),
                                  "largest_client": max(map(len, groups)),
                                  "clients_per_class": (counts > 0).sum(axis=0).tolist()}))


if __name__ == "__main__":
    main()
