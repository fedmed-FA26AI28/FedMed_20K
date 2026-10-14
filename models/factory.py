import random
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

from models.cnn import (
    CNN,
    SimpleCNN,
    get_parameters,
    set_parameters,
    count_parameters,
    get_bn_mask,
    get_buffer_mask,
    get_var_buffer_mask,
    set_parameters_fedbn,
    is_bn_key,
)
from models.baseline import BaselineCNN, ClassicCNN, PaperCNN


def set_seed(seed: int = 42) -> None:
    """Set global random seed across Python, NumPy, and PyTorch for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model(
    model_name: str = "cnn",
    num_classes: int = 8,
    in_channels: int = 3,
    input_size: Tuple[int, int] = (28, 28),
) -> nn.Module:
    """Factory function to instantiate models by name.
    
    Supported options:
      - 'cnn' (default): Modern lightweight CNN with BatchNorm and Global Average Pooling (~20K parameters).
      - 'baseline' / 'paper' / 'classic_cnn': Classic 4-layer CNN from FedSLD paper (Luo & Wu 2022) without BatchNorm (~1.62M parameters).
    """
    key = str(model_name).lower().strip().replace("-", "_")
    if key in ("baseline", "paper", "classic", "classic_cnn", "baseline_cnn"):
        return BaselineCNN(
            num_classes=num_classes,
            in_channels=in_channels,
            input_size=input_size,
        )
    elif key in ("cnn", "simple_cnn", "modern_cnn", "fedmedai"):
        return CNN(num_classes=num_classes)
    else:
        raise ValueError(
            f"Unknown model name '{model_name}'. Available options: 'cnn', 'baseline'."
        )
