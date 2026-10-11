"""Trien khai cac thuat toan chia nho dataset (IID va non-IID qua phan phoi Dirichlet, Pathological, Practical) cho cac client."""
import os
import sys
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from typing import List, Dict
from medmnist import INFO

# Duong dan mac dinh den file config
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


# ─────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────

def resolve_device_type(client_id: int) -> str:
    hw_path = _CONFIG_DIR / "jetson.yaml"
    with open(hw_path, "r", encoding="utf-8") as f:
        hw_cfg = yaml.safe_load(f)

    registry = hw_cfg.get("client_registry", {})
    if client_id not in registry:
        raise ValueError(
            f"client_id={client_id} khong co trong client_registry cua jetson.yaml. "
            f"Cac ID hop le: {sorted(registry.keys())}"
        )
    return registry[client_id]


def load_hardware_config(client_id: int = None, device_type: str = None) -> Dict:
    hw_path = _CONFIG_DIR / "jetson.yaml"
    with open(hw_path, "r", encoding="utf-8") as f:
        hw_cfg = yaml.safe_load(f)

    resolved_label = None
    if device_type:
        resolved_label = str(device_type).strip()
    elif client_id is not None:
        resolved_label = resolve_device_type(client_id)
    else:
        exp_path = _CONFIG_DIR / "experiment.yaml"
        if exp_path.exists():
            with open(exp_path, "r", encoding="utf-8") as f:
                exp_cfg = yaml.safe_load(f)
            resolved_label = exp_cfg.get("device_type", "pc")
        else:
            resolved_label = "pc"

    label_lower = resolved_label.lower()
    if resolved_label in hw_cfg:
        matched_hw = dict(hw_cfg[resolved_label])
    elif label_lower in hw_cfg:
        matched_hw = dict(hw_cfg[label_lower])
    elif label_lower.startswith("pc"):
        matched_hw = dict(hw_cfg.get("pc", {}))
    elif "orin" in label_lower:
        matched_hw = dict(hw_cfg.get("jetson_orin", {}))
    elif "nano" in label_lower:
        matched_hw = dict(hw_cfg.get("jetson_nano", {}))
    else:
        matched_hw = dict(hw_cfg.get("pc", {"batch_size": 32, "num_workers": 2, "pin_memory": True}))

    matched_hw["device_type"] = resolved_label
    return matched_hw


# ─────────────────────────────────────────────────────────────
# Partition
# ─────────────────────────────────────────────────────────────

def dirichlet_partition(dataset, num_clients: int, alpha: float, seed: int = 42) -> List[List[int]]:
    np.random.seed(seed)
    labels = np.array([dataset[i][1] for i in range(len(dataset))]).squeeze()
    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        np.random.shuffle(class_indices)
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        proportions = (proportions * len(class_indices)).astype(int)
        diff = len(class_indices) - proportions.sum()
        proportions[-1] += diff
        start = 0
        for cid, count in enumerate(proportions):
            end = start + count
            client_indices[cid].extend(class_indices[start:end].tolist())
            start = end
    return client_indices


def pathological_partition(dataset, num_clients: int, seed: int = 42) -> List[List[int]]:
    """Pathological Non-IID from FedSLD paper: Each client gets 2 random classes."""
    np.random.seed(seed)
    labels = np.array([dataset[i][1] for i in range(len(dataset))]).squeeze()
    num_classes = len(np.unique(labels))
    
    # Assign 2 random classes to each client
    # Ensure every class is assigned to at least one client to prevent dropped samples
    max_attempts = 1000
    for attempt in range(max_attempts):
        client_classes = [np.random.choice(num_classes, 2, replace=False) for _ in range(num_clients)]
        class_to_clients = {c: [] for c in range(num_classes)}
        for cid, classes in enumerate(client_classes):
            for c in classes:
                class_to_clients[c].append(cid)
        if all(len(clients) > 0 for clients in class_to_clients.values()):
            break
    else:
        # Fallback: force assign each class to at least one client
        client_classes = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            client_classes[c % num_clients].append(c)
        for cid in range(num_clients):
            while len(client_classes[cid]) < 2:
                c = np.random.choice(num_classes)
                if c not in client_classes[cid]:
                    client_classes[cid].append(c)
        class_to_clients = {c: [] for c in range(num_classes)}
        for cid, classes in enumerate(client_classes):
            for c in classes:
                class_to_clients[c].append(cid)
            
    client_indices = [[] for _ in range(num_clients)]
    
    for class_id in range(num_classes):
        assigned_clients = class_to_clients[class_id]
            
        class_indices = np.where(labels == class_id)[0]
        np.random.shuffle(class_indices)
        
        # Divide samples randomly among assigned clients
        proportions = np.random.dirichlet(np.ones(len(assigned_clients)))
        proportions = (proportions * len(class_indices)).astype(int)
        diff = len(class_indices) - proportions.sum()
        proportions[-1] += diff
        
        start = 0
        for idx, count in enumerate(proportions):
            end = start + count
            cid = assigned_clients[idx]
            client_indices[cid].extend(class_indices[start:end].tolist())
            start = end
            
    return client_indices


def practical_partition(dataset, num_clients: int, seed: int = 42) -> List[List[int]]:
    """Practical Non-IID from FedSLD paper: 12 shards per class (10x1%, 1x10%, 1x80%)."""
    np.random.seed(seed)
    labels = np.array([dataset[i][1] for i in range(len(dataset))]).squeeze()
    num_classes = len(np.unique(labels))
    
    if num_clients != 12:
        raise ValueError("practical_partition requires exactly 12 clients according to the FedSLD paper.")
        
    client_indices = [[] for _ in range(num_clients)]
    
    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        np.random.shuffle(class_indices)
        N = len(class_indices)
        
        # Shard counts: ten 1%, one 10%, one 80%
        shard_props = [0.01]*10 + [0.1, 0.8]
        shard_counts = [int(p * N) for p in shard_props]
        
        # Fix rounding errors
        diff = N - sum(shard_counts)
        shard_counts[-1] += diff  # add remainder to the 80% shard
        
        # Shuffle shards and assign one to each client
        shard_indices = list(range(12))
        np.random.shuffle(shard_indices)
        
        start = 0
        for cid, s_idx in enumerate(shard_indices):
            count = shard_counts[s_idx]
            end = start + count
            client_indices[cid].extend(class_indices[start:end].tolist())
            start = end
            
    return client_indices


def save_partition(client_indices: List[List[int]], dataset,
                   dataset_name: str, partition_mode: str, seed: int, num_clients: int,
                   save_dir: str = "data/partitions") -> Path:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    labels = np.array([dataset[i][1] for i in range(len(dataset))]).squeeze()
    num_classes = len(np.unique(labels))

    clients_data = {}
    for client_id, indices in enumerate(client_indices):
        client_labels = labels[indices]
        class_dict = {
            str(c): int(np.sum(client_labels == c))
            for c in range(num_classes)
        }
        clients_data[str(client_id)] = {
            "num_samples": len(indices),
            "indices": indices,
            "class_distribution": class_dict
        }

    result = {
        "dataset": dataset_name,
        "partition_mode": partition_mode,
        "seed": seed,
        "num_clients": num_clients,
        "total_samples": sum(len(idx) for idx in client_indices),
        "clients": clients_data
    }

    filename = f"partition_{dataset_name}_{partition_mode}_seed{seed}_clients{num_clients}.json"
    filepath = save_path / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    print(f"Partition saved: {filepath}")
    return filepath


def load_partition(json_path: str) -> List[List[int]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    num_clients = data["num_clients"]
    client_indices = []
    for i in range(num_clients):
        client_indices.append(data["clients"][str(i)]["indices"])
    return client_indices


# ─────────────────────────────────────────────────────────────
# DataLoader
# ─────────────────────────────────────────────────────────────

def get_client_dataloader(dataset, client_indices: List[int],
                          client_id: int = None,
                          device_type: str = None,
                          batch_size: int = None,
                          shuffle: bool = True,
                          num_workers: int = None,
                          pin_memory: bool = None) -> DataLoader:
    hw_cfg      = load_hardware_config(client_id=client_id, device_type=device_type)
    batch_size  = batch_size  if batch_size  is not None else hw_cfg["batch_size"]
    
    if num_workers is None:
        num_workers = 0 if sys.platform == "win32" else hw_cfg.get("num_workers", 0)
        
    if pin_memory is None:
        pin_memory  = hw_cfg.get("pin_memory", False)

    subset = Subset(dataset, client_indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def get_all_client_dataloaders(dataset, partition: List[List[int]],
                               batch_size: int = None,
                               num_workers: int = None,
                               pin_memory: bool = None) -> List[DataLoader]:
    exp_path = _CONFIG_DIR / "experiment.yaml"
    with open(exp_path, "r", encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)
    
    active_clients = exp_cfg.get("active_clients")
    if not active_clients or len(active_clients) != len(partition):
        active_clients = list(range(len(partition)))

    loaders = []
    for i, indices in enumerate(partition):
        cid = active_clients[i]
        loader = get_client_dataloader(dataset, indices,
                                       client_id=cid,
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       pin_memory=pin_memory)
        loaders.append(loader)
    return loaders


# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────

def visualize_partition(partition: List[List[int]], dataset,
                        dataset_name: str, partition_mode: str, num_clients: int,
                        save_path: str = None):
    labels = np.array([dataset[i][1] for i in range(len(dataset))]).squeeze()
    num_classes = len(np.unique(labels))
    
    # Lấy class names tu MedMNIST INFO
    data_flag = dataset_name.lower()
    if data_flag in INFO:
        label_dict = INFO[data_flag]['label']
        class_names = [label_dict[str(i)] for i in range(num_classes)]
    else:
        class_names = [f"Class {i}" for i in range(num_classes)]

    dist_matrix = np.zeros((num_clients, num_classes), dtype=int)
    for client_id, indices in enumerate(partition):
        client_labels = labels[indices]
        for c in range(num_classes):
            dist_matrix[client_id, c] = int(np.sum(client_labels == c))

    fig, ax = plt.subplots(figsize=(max(8, num_clients * 1.2), 5))
    x = np.arange(num_clients)
    bottom = np.zeros(num_clients)
    colors = plt.cm.Set3(np.linspace(0, 1, num_classes))

    for c in range(num_classes):
        ax.bar(x, dist_matrix[:, c], bottom=bottom,
               label=class_names[c], color=colors[c])
        bottom += dist_matrix[:, c]

    ax.set_title(f"Data Distribution | {dataset_name} | {partition_mode} | {num_clients} Clients")
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Number of Samples")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i}" for i in range(num_clients)])
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {save_path}")
    plt.close()