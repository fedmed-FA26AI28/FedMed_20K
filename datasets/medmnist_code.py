"""Data loading and preprocessing utilities for MedMNIST datasets (e.g. BloodMNIST, PathMNIST, OrganAMNIST)."""

import torch
import torchvision.transforms as transforms
from medmnist import INFO
from torch.utils.data import DataLoader


def get_medmnist_datasets(dataset_name: str = "bloodmnist", download: bool = True):
    """Load and return MedMNIST dataset splits.

    Args:
        dataset_name (str): Name of the dataset (e.g. 'bloodmnist', 'pathmnist', 'organamnist').
        download (bool): Automatically download data if not present locally.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, num_classes)
    """
    data_flag = dataset_name.lower()
    if data_flag not in INFO:
        raise ValueError(f"Dataset {data_flag} not found in MedMNIST INFO.")
        
    info = INFO[data_flag]
    num_classes = len(info['label'])
    n_channels = info.get('n_channels', 3)
    
    # Deferred import to prevent circular import with module namespace
    import medmnist as _medmnist
    DataClass = getattr(_medmnist, info['python_class'])

    # Determine transforms based on channels
    transform_list = [transforms.ToTensor()]
    
    # If the dataset is 1-channel (like organamnist), convert to 3-channel
    # so that our CNN architectures designed for 3-channel input work seamlessly.
    if n_channels == 1:
        # Grayscale to RGB
        transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
        
    transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    
    data_transform = transforms.Compose(transform_list)

    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    val_dataset = DataClass(split='val', transform=data_transform, download=download)
    test_dataset = DataClass(split='test', transform=data_transform, download=download)

    return train_dataset, val_dataset, test_dataset, num_classes


def get_medmnist_dataloaders(dataset_name: str = "bloodmnist", batch_size: int = 32, download: bool = True, num_workers: int = None):
    """Return PyTorch DataLoaders for centralized baseline training."""
    if num_workers is None:
        import sys
        num_workers = 0 if sys.platform == "win32" else 2
    train_dataset, val_dataset, test_dataset, num_classes = get_medmnist_datasets(dataset_name=dataset_name, download=download)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, num_classes
