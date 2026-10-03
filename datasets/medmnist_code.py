"""Data loading and preprocessing utilities for MedMNIST datasets (e.g. BloodMNIST)."""

import torch
import torchvision.transforms as transforms
from medmnist import INFO
from torch.utils.data import DataLoader


def get_bloodmnist_datasets(batch_size: int = 32, download: bool = True):
    """Load and return BloodMNIST dataset splits.

    Args:
        batch_size (int): Batch size.
        download (bool): Automatically download data if not present locally.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, num_classes)
    """
    data_flag = 'bloodmnist'
    info = INFO[data_flag]
    num_classes = len(info['label'])
    
    # Deferred import to prevent circular import with module namespace
    import medmnist as _medmnist
    DataClass = getattr(_medmnist, info['python_class'])

    # Standard normalization transform for 3-channel BloodMNIST (28x28x3)
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    val_dataset = DataClass(split='val', transform=data_transform, download=download)
    test_dataset = DataClass(split='test', transform=data_transform, download=download)

    return train_dataset, val_dataset, test_dataset, num_classes


def get_bloodmnist_dataloaders(batch_size: int = 32, download: bool = True, num_workers: int = None):
    """Return PyTorch DataLoaders for centralized baseline training."""
    if num_workers is None:
        import sys
        num_workers = 0 if sys.platform == "win32" else 2
    train_dataset, val_dataset, test_dataset, num_classes = get_bloodmnist_datasets(download=download)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, num_classes
