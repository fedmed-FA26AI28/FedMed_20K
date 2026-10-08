import torch
import torch.nn as nn
from typing import List
import numpy as np


class CNN(nn.Module):
    def __init__(self, num_classes: int = 8):
        super().__init__()

        # Conv Block 1: Input 3 x 28 x 28 -> Output 32 x 28 x 28
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # Conv Block 2: Input 32 x 28 x 28 -> Output 64 x 28 x 28
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Max Pooling: 64 x 28 x 28 -> 64 x 14 x 14
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global Average Pooling: 64 x 14 x 14 -> 64 x 1 x 1
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layer: 64 -> num_classes
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.pool(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class GroupNormCNN(CNN):
    """Same tiny CNN, with per-example normalization and no running BN state."""

    def __init__(self, num_classes: int = 8):
        super().__init__(num_classes=num_classes)
        self.conv_block1[1] = nn.GroupNorm(8, 32)
        self.conv_block2[1] = nn.GroupNorm(8, 64)


class MobileNetSmall(nn.Module):
    """Untrained MobileNetV3-Small with an exposed linear FL classifier head."""
    def __init__(self, num_classes: int = 8):
        super().__init__()
        from torchvision.models import mobilenet_v3_small
        self.backbone = mobilenet_v3_small(weights=None)
        features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Identity()
        self.fc = nn.Linear(features, num_classes)

    def forward(self, x):
        return self.fc(self.backbone(x))


def build_model(name: str = "legacy", num_classes: int = 8):
    """Build a shared architecture for the full and 5K training budgets."""
    factories = {
        "legacy": lambda: CNN(num_classes=num_classes),
        "tiny_cnn": lambda: CNN(num_classes=num_classes),
        "tiny_cnn_gn": lambda: GroupNormCNN(num_classes=num_classes),
        "mobilenet_v3_small": lambda: MobileNetSmall(num_classes=num_classes),
    }
    if name not in factories:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(factories)}")
    return factories[name]()


def count_parameters(model: nn.Module):
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return total_parameters, trainable_parameters


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """Trích xuất trọng số mô hình PyTorch thành danh sách các mảng NumPy cho Flower."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Nạp danh sách trọng số NumPy vào mô hình PyTorch cho Flower."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


# Alias tương thích ngược cho các script đang dùng SimpleCNN
SimpleCNN = CNN


if __name__ == "__main__":
    # Chạy kiểm thử kiến trúc mô hình
    model = CNN(num_classes=8)
    dummy_input = torch.randn(4, 3, 28, 28)
    output = model(dummy_input)

    total_params, trainable_params = count_parameters(model)
    print("=== MODEL TEST: CNN ===")
    print(f"Input shape : {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    assert output.shape == (4, 8), "Error: Output shape is not (4, 8)!"
    print("Test passed successfully!")
