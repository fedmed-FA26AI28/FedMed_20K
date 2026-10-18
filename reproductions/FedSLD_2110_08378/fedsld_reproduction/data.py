"""MedMNIST loading utilities for the reference experiments."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import ConcatDataset
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


def dataset_labels(dataset) -> np.ndarray:
    if isinstance(dataset, ConcatDataset):
        return np.concatenate([dataset_labels(part) for part in dataset.datasets])
    if not hasattr(dataset, "labels"):
        raise TypeError("dataset does not expose MedMNIST labels")
    return np.asarray(dataset.labels, dtype=np.int64).reshape(-1)


def load_medmnist(
    flag: str,
    root: str = "data",
    include_validation_in_train: bool = True,
    download: bool = True,
) -> DatasetBundle:
    if flag not in {"organamnist", "pathmnist", "bloodmnist"}:
        raise ValueError("dataset must be organamnist, pathmnist, or bloodmnist")
    import medmnist
    from medmnist import INFO

    info = INFO[flag]
    dataset_class = getattr(medmnist, info["python_class"])
    options = dict(root=root, transform=ToTensor(), download=download, size=28)
    train_part = dataset_class(split="train", **options)
    if include_validation_in_train:
        validation_part = dataset_class(split="val", **options)
        train = ConcatDataset([train_part, validation_part])
    else:
        train = train_part
    test = dataset_class(split="test", **options)
    return DatasetBundle(
        train=train,
        test=test,
        train_labels=dataset_labels(train),
        test_labels=dataset_labels(test),
        in_channels=int(info["n_channels"]),
        num_classes=len(info["label"]),
        image_size=28,
    )

