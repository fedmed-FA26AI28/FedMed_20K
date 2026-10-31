"""Model choices shared by all matched research methods."""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyCNNGN(nn.Module):
    """Small GroupNorm control used in earlier research experiments."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.MaxPool2d(2), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.fc(self.encoder(images))


class TinyCNNBN(TinyCNNGN):
    """Small CNN with globally aggregated BatchNorm statistics."""

    def __init__(self, num_classes: int):
        super().__init__(num_classes)
        self.encoder[1] = nn.BatchNorm2d(32)
        self.encoder[4] = nn.BatchNorm2d(64)


def make_model(num_classes: int, norm: str = "gn") -> nn.Module:
    if norm == "gn":
        return TinyCNNGN(num_classes)
    if norm == "bn":
        return TinyCNNBN(num_classes)
    if norm == "baseline":
        from models.baseline import BaselineCNN
        return BaselineCNN(num_classes=num_classes)
    if norm == "baseline_bn":
        from models.baseline_bn import BaselineBN
        return BaselineBN(num_classes=num_classes)
    raise ValueError(f"unknown model: {norm}")


def classifier_head(model: nn.Module) -> nn.Linear:
    """Return the final linear classifier for either research or team CNN."""
    head = getattr(model, "fc2", None)
    if head is None:
        head = getattr(model, "fc", None)
    if not isinstance(head, nn.Linear):
        raise TypeError("SCR requires a linear classifier named fc or fc2")
    return head
