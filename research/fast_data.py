"""Tensor-backed MedMNIST input path for single-process sequential clients."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class FastImageDataset(Dataset):
    """Expose raw uint8 images without per-sample PIL conversion."""

    def __init__(self, source, resolution: int):
        array = np.asarray(source.imgs)
        if array.dtype != np.uint8 or array.ndim not in (3, 4):
            raise ValueError("fast input requires uint8 2D MedMNIST images")
        self.images = torch.from_numpy(array)
        if array.ndim == 3:
            self.images = self.images.unsqueeze(1)
        else:
            self.images = self.images.permute(0, 3, 1, 2)
        self.labels = source.labels
        self.resolution = resolution

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], int(np.asarray(self.labels[index]).reshape(-1)[0])


def prepare_batch(images: torch.Tensor, resolution: int, training: bool) -> torch.Tensor:
    """Apply the same crop/flip distribution as the PIL path, then normalize."""
    if images.dtype != torch.uint8 or images.ndim != 4:
        raise ValueError("expected BCHW uint8 images")
    images = images.float().div_(255)
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    if training:
        batch, channels, height, width = images.shape
        padded = F.pad(images, (2, 2, 2, 2), mode="reflect")
        top = torch.randint(0, 5, (batch,), device=images.device)
        left = torch.randint(0, 5, (batch,), device=images.device)
        cols = (left[:, None] + torch.arange(width, device=images.device)).view(batch, 1, 1, width)
        cols = cols.expand(batch, channels, height + 4, width)
        cropped_width = padded.gather(3, cols)
        rows = (top[:, None] + torch.arange(height, device=images.device)).view(batch, 1, height, 1)
        rows = rows.expand(batch, channels, height, width)
        images = cropped_width.gather(2, rows)
        flip = torch.rand(batch, 1, 1, 1, device=images.device) < 0.5
        images = torch.where(flip, images.flip(-1), images)
    if images.shape[-1] != resolution or images.shape[-2] != resolution:
        images = F.interpolate(images, size=(resolution, resolution),
                               mode="bilinear", align_corners=False)
    return images.mul_(2).sub_(1)
