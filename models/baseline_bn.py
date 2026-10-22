import torch
import torch.nn as nn
from typing import List, Tuple
import numpy as np


class BaselineBN(nn.Module):
    """Classic 4-layer CNN model following the FedSLD paper (Luo & Wu, ISBI 2022 / McMahan et al., 2017),
    augmented with Batch Normalization (BN) layers.

    Architecture:
      - Conv 1: 5x5 Conv (in_channels -> 32) + BatchNorm2d(32) + ReLU + 2x2 MaxPool
      - Conv 2: 5x5 Conv (32 -> 64) + BatchNorm2d(64) + ReLU + 2x2 MaxPool
      - FC 1  : Linear (flattened_dim -> 500) + [Optional BatchNorm1d(500)] + ReLU
      - FC 2  : Linear (500 -> num_classes)
    """

    def __init__(
        self,
        num_classes: int = 8,
        in_channels: int = 3,
        input_size: Tuple[int, int] = (28, 28),
        padding: int = 0,
        include_fc_bn: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.padding = padding
        self.include_fc_bn = include_fc_bn

        # Conv Block 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=padding)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=padding)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Compute flattened dimension dynamically based on input size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size[0], input_size[1])
            x = self.pool1(self.relu1(self.bn1(self.conv1(dummy))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            flattened_dim = x.view(1, -1).size(1)

        self.flattened_dim = flattened_dim
        self.fc1 = nn.Linear(flattened_dim, 500)
        if self.include_fc_bn:
            self.bn3 = nn.BatchNorm1d(500)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        if self.include_fc_bn:
            x = self.bn3(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return total_parameters, trainable_parameters


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract PyTorch model weights as a list of NumPy ndarrays for Flower."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Load a list of NumPy ndarrays into PyTorch model weights for Flower."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


# Aliases for convenience
BaselineCNN_BN = BaselineBN
BaselineCNNBN = BaselineBN
ClassicBN = BaselineBN
PaperBN = BaselineBN


if __name__ == "__main__":
    model = BaselineBN(num_classes=9, in_channels=3, input_size=(28, 28))
    dummy_input = torch.randn(4, 3, 28, 28)
    output = model(dummy_input)

    total_params, trainable_params = count_parameters(model)
    print("=== MODEL TEST: BaselineBN (Baseline CNN with added BN layers) ===")
    print(f"Input shape : {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    assert output.shape == (4, 9), "Error: Output shape is not (4, 9)!"
    print("Test passed successfully!")
