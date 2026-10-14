"""Data loading and preprocessing utilities for MedMNIST datasets (e.g. BloodMNIST, PathMNIST, OrganAMNIST)."""

import torch
import torchvision.transforms as transforms
from medmnist import INFO
from torch.utils.data import DataLoader


def get_medmnist_datasets(
    dataset_name: str = "bloodmnist",
    download: bool = True,
    augment: bool = True,
    random_crop: bool = True,
    random_flip: bool = True,
    crop_padding: int = 2,
):
    """Load and return MedMNIST dataset splits.

    Args:
        dataset_name (str): Name of the dataset (e.g. 'bloodmnist', 'pathmnist', 'organamnist').
        download (bool): Automatically download data if not present locally.
        augment (bool): Master toggle for training set data augmentation (default: True).
        random_crop (bool): Apply RandomCrop with reflection padding to train set (default: True).
        random_flip (bool): Apply RandomHorizontalFlip to train set (default: True).
        crop_padding (int): Padding size for RandomCrop (default: 2 for 28x28 images).

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

    # 1. Validation & Test transform: strictly deterministic (no augmentation)
    val_test_list = [transforms.ToTensor()]
    if n_channels == 1:
        val_test_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
    val_test_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    val_test_transform = transforms.Compose(val_test_list)

    # 2. Training transform: data augmentation (crop + flip) when enabled
    train_list = []
    if augment:
        if random_crop:
            train_list.append(transforms.RandomCrop(28, padding=crop_padding, padding_mode='reflect'))
        if random_flip:
            train_list.append(transforms.RandomHorizontalFlip(p=0.5))
            
    train_list.append(transforms.ToTensor())
    if n_channels == 1:
        train_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
    train_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    train_transform = transforms.Compose(train_list)

    train_dataset = DataClass(split='train', transform=train_transform, download=download)
    val_dataset = DataClass(split='val', transform=val_test_transform, download=download)
    test_dataset = DataClass(split='test', transform=val_test_transform, download=download)

    return train_dataset, val_dataset, test_dataset, num_classes


def get_medmnist_dataloaders(
    dataset_name: str = "bloodmnist",
    batch_size: int = 32,
    download: bool = True,
    num_workers: int = None,
    augment: bool = True,
    random_crop: bool = True,
    random_flip: bool = True,
    crop_padding: int = 2,
):
    """Return PyTorch DataLoaders for centralized baseline training."""
    if num_workers is None:
        import sys
        num_workers = 0 if sys.platform == "win32" else 2
    train_dataset, val_dataset, test_dataset, num_classes = get_medmnist_datasets(
        dataset_name=dataset_name,
        download=download,
        augment=augment,
        random_crop=random_crop,
        random_flip=random_flip,
        crop_padding=crop_padding,
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, num_classes


# Backward-compatible aliases for legacy scripts
get_bloodmnist_datasets = lambda download=True, **kwargs: get_medmnist_datasets("bloodmnist", download=download, **kwargs)
get_bloodmnist_dataloaders = lambda batch_size=32, download=True, num_workers=None, **kwargs: get_medmnist_dataloaders("bloodmnist", batch_size=batch_size, download=download, num_workers=num_workers, **kwargs)
