import torch
import torch.nn as nn
from typing import List
import numpy as np


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
