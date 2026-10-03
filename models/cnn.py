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


def count_parameters(model: nn.Module):
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


def is_bn_key(model: nn.Module, key: str) -> bool:
    """Check if a state_dict key belongs to a BatchNorm layer."""
    submod_name = key.rsplit(".", 1)[0] if "." in key else ""
    try:
        submod = model.get_submodule(submod_name) if submod_name else model
        return isinstance(submod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
    except Exception:
        key_lower = key.lower()
        return "bn" in key_lower or "batchnorm" in key_lower


def get_bn_mask(model: nn.Module) -> List[bool]:
    """Return a boolean mask where True indicates a BatchNorm parameter or buffer."""
    return [is_bn_key(model, k) for k in model.state_dict().keys()]


def get_buffer_mask(model: nn.Module) -> List[bool]:
    """Return a boolean mask where True indicates a non-trainable buffer (e.g. running_mean, running_var, num_batches_tracked)."""
    named_buffers = set(dict(model.named_buffers()).keys())
    return [k in named_buffers for k in model.state_dict().keys()]


def get_var_buffer_mask(model: nn.Module) -> List[bool]:
    """Return a boolean mask where True indicates a variance buffer (e.g. running_var)."""
    return ["running_var" in k for k in model.state_dict().keys()]


def set_parameters_fedbn(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Load only non-BatchNorm parameters into the model, preserving local BN states."""
    current_state = model.state_dict()
    new_state = {}
    for (k, v), param_array in zip(current_state.items(), parameters):
        if is_bn_key(model, k):
            # Preserve local client BN parameters and running statistics
            new_state[k] = v
        else:
            # Overwrite non-BN layers with global parameters
            new_state[k] = torch.as_tensor(param_array)
    model.load_state_dict(new_state, strict=True)


# Backward compatibility alias for scripts referencing SimpleCNN
SimpleCNN = CNN


if __name__ == "__main__":
    # Test model architecture forward pass and parameter count
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
