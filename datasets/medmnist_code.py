"""Data loading and preprocessing utilities for MedMNIST datasets (e.g. BloodMNIST, PathMNIST, OrganAMNIST)."""

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
import torchvision.transforms as transforms
from medmnist import INFO
from medmnist.info import DEFAULT_ROOT
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def get_medmnist_datasets(
    dataset_name: str = "bloodmnist",
    download: bool = True,
    augment: bool = True,
    random_crop: bool = True,
    random_flip: bool = True,
    crop_padding: int = 2,
    load_train: bool = True,
    load_val: bool = True,
    load_test: bool = True,
):
    """Load and return MedMNIST dataset splits.

    Args:
        dataset_name (str): Name of the dataset (e.g. 'bloodmnist', 'pathmnist', 'organamnist').
        download (bool): Automatically download data if not present locally.
        augment (bool): Master toggle for training set data augmentation (default: True).
        random_crop (bool): Apply RandomCrop with reflection padding to train set (default: True).
        random_flip (bool): Apply RandomHorizontalFlip to train set (default: True).
        crop_padding (int): Padding size for RandomCrop (default: 2 for 28x28 images).
        load_train/load_val/load_test (bool): Load only splits needed by the caller.

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

    train_transform, val_test_transform = _make_transforms(
        n_channels, augment, random_crop, random_flip, crop_padding
    )

    train_dataset = DataClass(split='train', transform=train_transform, download=download) if load_train else None
    val_dataset = DataClass(split='val', transform=val_test_transform, download=download) if load_val else None
    test_dataset = DataClass(split='test', transform=val_test_transform, download=download) if load_test else None

    return train_dataset, val_dataset, test_dataset, num_classes


def _make_transforms(n_channels, augment, random_crop, random_flip, crop_padding):
    # Validation and test transforms are deterministic.
    val_test_list = [transforms.ToTensor()]
    if n_channels == 1:
        val_test_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
    val_test_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    val_test_transform = transforms.Compose(val_test_list)

    # Training augmentation matches the regular MedMNIST loader.
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

    return train_transform, val_test_transform


class _IndexedMedMNISTDataset(Dataset):
    """The same 2D image/label behavior as MedMNIST, with only selected rows."""

    def __init__(self, imgs, labels, transform, info, split):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.split = split
        self.info = dict(info)
        self.info['n_samples'] = dict(info['n_samples'])
        self.info['n_samples'][split] = len(imgs)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = Image.fromarray(self.imgs[index])
        target = self.labels[index].astype(int)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def _read_indexed_images(npz_path, split, indices):
    """Stream selected image rows from a compressed NPZ without expanding the split."""
    selected = np.asarray(indices, dtype=np.intp)
    if selected.ndim != 1:
        raise ValueError("Client indices must be one-dimensional")

    with ZipFile(npz_path) as archive, archive.open(f'{split}_images.npy') as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f'Unsupported NPY version in {npz_path}: {version}')
        if fortran_order or dtype.hasobject or len(shape) < 2:
            raise ValueError('Indexed MedMNIST loading requires a C-order numeric image array')
        if np.any(selected < 0) or np.any(selected >= shape[0]):
            raise IndexError('Client index is outside the MedMNIST split')

        row_bytes = int(np.prod(shape[1:])) * dtype.itemsize
        images = np.empty((len(selected), *shape[1:]), dtype=dtype)
        current_row = 0
        last_source = -1
        last_dest = -1
        for dest in np.argsort(selected, kind='stable'):
            source = int(selected[dest])
            if source == last_source:
                images[dest] = images[last_dest]
                continue
            bytes_to_skip = (source - current_row) * row_bytes
            while bytes_to_skip:
                chunk = stream.read(min(bytes_to_skip, 1024 * 1024))
                if not chunk:
                    raise EOFError(f'Incomplete image array in {npz_path}')
                bytes_to_skip -= len(chunk)
            row = stream.read(row_bytes)
            if len(row) != row_bytes:
                raise EOFError(f'Incomplete image array in {npz_path}')
            images[dest] = np.frombuffer(row, dtype=dtype).reshape(shape[1:])
            current_row = source + 1
            last_source = source
            last_dest = dest
    return images, selected


def get_medmnist_client_datasets(
    dataset_name: str,
    train_indices,
    test_indices=None,
    augment: bool = True,
    random_crop: bool = True,
    random_flip: bool = True,
    crop_padding: int = 2,
):
    """Load client shards from the cached NPZ without allocating the full train split.

    The simulation runner verifies/downloads the archive once before clients launch.
    Passing test_indices=None loads the full test split for a global client test.
    """
    data_flag = dataset_name.lower()
    if data_flag not in INFO:
        raise ValueError(f'Dataset {data_flag} not found in MedMNIST INFO.')
    info = INFO[data_flag]
    npz_path = Path(DEFAULT_ROOT) / f'{data_flag}.npz'
    if not npz_path.is_file():
        raise FileNotFoundError(f'MedMNIST archive is missing: {npz_path}')

    train_transform, test_transform = _make_transforms(
        info.get('n_channels', 3), augment, random_crop, random_flip, crop_padding
    )
    train_images, train_selected = _read_indexed_images(npz_path, 'train', train_indices)
    with np.load(npz_path, allow_pickle=False) as archive:
        train_labels = archive['train_labels'][train_selected]
    train_dataset = _IndexedMedMNISTDataset(train_images, train_labels, train_transform, info, 'train')

    if test_indices is None:
        import medmnist as _medmnist
        DataClass = getattr(_medmnist, info['python_class'])
        test_dataset = DataClass(split='test', transform=test_transform, download=False)
    else:
        test_images, test_selected = _read_indexed_images(npz_path, 'test', test_indices)
        with np.load(npz_path, allow_pickle=False) as archive:
            test_labels = archive['test_labels'][test_selected]
        test_dataset = _IndexedMedMNISTDataset(test_images, test_labels, test_transform, info, 'test')

    return train_dataset, test_dataset, len(info['label'])


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
