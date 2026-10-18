"""MedMNIST loading utilities for the reference experiments."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset
from torchvision.transforms import ToTensor


@dataclass(frozen=True)
class DatasetBundle:
    train: torch.utils.data.Dataset
    test: torch.utils.data.Dataset
    train_labels: np.ndarray
    test_labels: np.ndarray
    in_channels: int
    num_classes: int
    image_size: int
    split_mode: str
    test_fraction: float


def dataset_labels(dataset) -> np.ndarray:
    if isinstance(dataset, ConcatDataset):
        return np.concatenate([dataset_labels(part) for part in dataset.datasets])
    if not hasattr(dataset, "labels"):
        raise TypeError("dataset does not expose MedMNIST labels")
    return np.asarray(dataset.labels, dtype=np.int64).reshape(-1)


def stratified_split_indices(labels, test_fraction: float, seed: int):
    """Split every class reproducibly while matching the requested total size."""
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if y.size == 0 or not 0 < test_fraction < 1:
        raise ValueError("a non-empty label array and test fraction in (0, 1) are required")
    classes, class_counts = np.unique(y, return_counts=True)
    if np.any(class_counts < 2):
        raise ValueError("every class needs at least two examples for a stratified split")

    desired_test_total = int(round(y.size * test_fraction))
    raw_test_counts = class_counts * test_fraction
    test_counts = np.floor(raw_test_counts).astype(np.int64)
    test_counts = np.clip(test_counts, 1, class_counts - 1)
    remainder = desired_test_total - int(test_counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw_test_counts - np.floor(raw_test_counts)), kind="stable")
        for class_position in order:
            if remainder == 0:
                break
            if test_counts[class_position] < class_counts[class_position] - 1:
                test_counts[class_position] += 1
                remainder -= 1
    elif remainder < 0:
        order = np.argsort(raw_test_counts - np.floor(raw_test_counts), kind="stable")
        for class_position in order:
            if remainder == 0:
                break
            if test_counts[class_position] > 1:
                test_counts[class_position] -= 1
                remainder += 1
    if remainder != 0:
        raise RuntimeError("could not allocate the requested stratified test size")

    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []
    for class_id, test_count in zip(classes, test_counts):
        indices = np.flatnonzero(y == class_id)
        rng.shuffle(indices)
        test_indices.extend(indices[:test_count].tolist())
        train_indices.extend(indices[test_count:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return train_indices, test_indices


def load_medmnist(
    flag: str,
    root: str = "data",
    include_validation_in_train: bool = True,
    download: bool = True,
    split_mode: str = "official",
    split_seed: int = 42,
    stratified_test_fraction: float = None,
) -> DatasetBundle:
    if flag not in {"organamnist", "pathmnist", "bloodmnist"}:
        raise ValueError("dataset must be organamnist, pathmnist, or bloodmnist")
    if split_mode not in {"official", "stratified"}:
        raise ValueError("split_mode must be official or stratified")
    import medmnist
    from medmnist import INFO

    data_root = Path(root).expanduser()
    try:
        data_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise RuntimeError(f"could not create MedMNIST data root: {data_root}") from error

    info = INFO[flag]
    dataset_class = getattr(medmnist, info["python_class"])
    options = dict(root=str(data_root), transform=ToTensor(), download=download, size=28)
    train_part = dataset_class(split="train", **options)
    validation_part = dataset_class(split="val", **options)
    test_part = dataset_class(split="test", **options)

    if split_mode == "stratified":
        if not include_validation_in_train:
            raise ValueError("stratified mode always pools official train, val, and test")
        combined = ConcatDataset([train_part, validation_part, test_part])
        combined_labels = dataset_labels(combined)
        test_fraction = (
            len(test_part) / len(combined)
            if stratified_test_fraction is None
            else float(stratified_test_fraction)
        )
        train_indices, test_indices = stratified_split_indices(
            combined_labels, test_fraction, split_seed
        )
        train = Subset(combined, train_indices)
        test = Subset(combined, test_indices)
        train_labels = combined_labels[train_indices]
        test_labels = combined_labels[test_indices]
    else:
        train = (
            ConcatDataset([train_part, validation_part])
            if include_validation_in_train
            else train_part
        )
        test = test_part
        train_labels = dataset_labels(train)
        test_labels = dataset_labels(test)
        test_fraction = len(test) / (len(train) + len(test))

    return DatasetBundle(
        train=train,
        test=test,
        train_labels=train_labels,
        test_labels=test_labels,
        in_channels=int(info["n_channels"]),
        num_classes=len(info["label"]),
        image_size=28,
        split_mode=split_mode,
        test_fraction=float(test_fraction),
    )

