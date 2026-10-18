"""Single-process synchronous FL used for the independent reconstruction."""

import copy
from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .loss import fedsld_loss


def aggregate_states(
    states: Iterable[Dict[str, torch.Tensor]], sample_counts: Iterable[int]
) -> OrderedDict:
    states = list(states)
    counts = np.asarray(list(sample_counts), dtype=np.float64)
    if not states or len(states) != len(counts):
        raise ValueError("one positive sample count is required for every state")
    if np.any(counts <= 0):
        raise ValueError("sample counts must be positive")
    weights = counts / counts.sum()
    result = OrderedDict()
    for key in states[0]:
        tensors = [state[key].detach().cpu() for state in states]
        if any(tensor.shape != tensors[0].shape for tensor in tensors):
            raise ValueError(f"state shape mismatch for {key}")
        value = sum(float(weight) * tensor.to(torch.float64) for weight, tensor in zip(weights, tensors))
        result[key] = value.to(tensors[0].dtype)
    return result


def train_client(
    global_model: torch.nn.Module,
    dataset,
    indices: List[int],
    algorithm: str,
    global_prior: torch.Tensor,
    device: torch.device,
    batch_size: int,
    local_epochs: int,
    learning_rate: float,
    proximal_mu: float,
    seed: int,
) -> Tuple[OrderedDict, float]:
    model = copy.deepcopy(global_model).to(device)
    model.train()
    reference = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    total_loss = 0.0
    total_examples = 0
    for _ in range(local_epochs):
        for images, labels in loader:
            images = images.to(device)
            targets = labels.reshape(-1).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if algorithm == "fedsld":
                loss = fedsld_loss(logits, targets, global_prior.to(device))
            else:
                loss = F.cross_entropy(logits, targets)
            if algorithm == "fedprox":
                proximal = sum(
                    torch.sum((local - global_value) ** 2)
                    for local, global_value in zip(model.parameters(), reference)
                )
                loss = loss + 0.5 * proximal_mu * proximal
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * targets.numel()
            total_examples += targets.numel()
    state = OrderedDict((key, value.detach().cpu().clone()) for key, value in model.state_dict().items())
    return state, total_loss / max(1, total_examples)

