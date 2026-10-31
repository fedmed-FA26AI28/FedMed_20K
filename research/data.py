"""Load official MedMNIST splits, caching new downloads inside the workspace."""

from __future__ import annotations

from pathlib import Path

import medmnist
from medmnist import INFO
from medmnist.info import DEFAULT_ROOT


REPO = Path(__file__).resolve().parents[1]
WORKSPACE_CACHE = REPO / "data" / "medmnist"


def archive_path(flag: str) -> Path:
    name = f"{flag}.npz"
    default = Path(DEFAULT_ROOT) / name
    return default if default.exists() else WORKSPACE_CACHE / name


def get_medmnist_datasets(dataset_name: str, *, download: bool = False,
                         augment: bool = False, load_train: bool = True,
                         load_val: bool = True, load_test: bool = True):
    """Return only requested official splits; transforms are set by the caller."""
    del augment  # Runner sets identical transforms after selecting the data path.
    flag = dataset_name.lower()
    if flag not in INFO:
        raise ValueError(f"unknown MedMNIST dataset: {dataset_name}")
    info = INFO[flag]
    root = archive_path(flag).parent
    if root == WORKSPACE_CACHE:
        root.mkdir(parents=True, exist_ok=True)
    data_class = getattr(medmnist, info["python_class"])
    def load(split: str):
        return data_class(split=split, transform=None, download=download, root=str(root))
    train = load("train") if load_train else None
    val = load("val") if load_val else None
    test = load("test") if load_test else None
    return train, val, test, len(info["label"])
