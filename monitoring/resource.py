"""Hardware resource monitoring utilities for FedMedAI.

Provides cross-platform hardware sampling (CPU, RAM, GPU VRAM)
and device-type resolution for heterogeneous edge experiments (PC, Jetson Orin, Jetson Nano).
"""

import sys
import psutil
import torch
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import yaml
except ImportError:
    yaml = None


def get_resource_usage() -> Dict[str, float]:
    """Sample current CPU, RAM, and GPU memory usage.

    Returns:
        Dict with keys:
            - cpu_percent: CPU utilization percentage (0-100)
            - ram_percent: Virtual memory utilization percentage (0-100)
            - ram_used_mb: Virtual memory used in megabytes
            - gpu_memory_allocated_mb: PyTorch allocated GPU memory in MB
            - gpu_memory_reserved_mb: PyTorch reserved GPU memory in MB
    """
    vm = psutil.virtual_memory()
    gpu_allocated = 0.0
    gpu_reserved = 0.0

    if torch.cuda.is_available():
        try:
            gpu_allocated = float(torch.cuda.memory_allocated() / (1024 ** 2))
            gpu_reserved = float(torch.cuda.memory_reserved() / (1024 ** 2))
        except Exception:
            gpu_allocated = 0.0
            gpu_reserved = 0.0

    return {
        "cpu_percent": float(psutil.cpu_percent(interval=None)),
        "ram_percent": float(vm.percent),
        "ram_used_mb": float(vm.used / (1024 ** 2)),
        "gpu_memory_allocated_mb": round(gpu_allocated, 2),
        "gpu_memory_reserved_mb": round(gpu_reserved, 2),
    }


def get_device_type(client_id: Optional[int] = None, override: Optional[str] = None) -> str:
    """Resolve device type ('pc', 'jetson_orin', 'jetson_nano', or custom label like 'PC1').

    If override is provided and non-empty, returns override directly.
    Otherwise tries to lookup client_id in configs/jetson.yaml client_registry.
    Falls back to 'pc' if client_id is None or unlisted.
    """
    if override:
        return str(override).strip()

    if client_id is None:
        return "pc"

    config_path = Path(__file__).resolve().parent.parent / "configs" / "jetson.yaml"
    if config_path.exists() and yaml is not None:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            registry = cfg.get("client_registry", {})
            if client_id in registry:
                return str(registry[client_id])
            if str(client_id) in registry:
                return str(registry[str(client_id)])
        except Exception:
            pass

    return "pc"
