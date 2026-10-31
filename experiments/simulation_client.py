"""Simulation Client Wrapper for Federated Learning Experiments.

Enforces serialized/pooled GPU execution across concurrent client subprocesses
during simulated multi-client runs on a single host.

IMPORTANT:
- client/client.py is the canonical entry point for physical edge hardware
  (Jetson Orin, Jetson Nano, PCs). This wrapper calls its training function.
- experiments/simulation_client.py is strictly used by experiments/run_simulation.py
  to prevent simultaneous GPU training on single machines. Worker processes
  still load PyTorch and datasets separately and require sufficient host RAM.
"""

import os
import sys
import time
from pathlib import Path

try:
    from filelock import FileLock
except ImportError:
    FileLock = None

import torch
import client.client as client_module

_orig_local_train = client_module._local_train
_orig_evaluate = client_module.FedMedAIClient.evaluate


def _get_cid_str() -> str:
    """Extract client id from CLI arguments for clear logging."""
    for i, a in enumerate(sys.argv):
        if a.startswith("--client_id="):
            return f"Client {a.split('=')[1]}"
        if a == "--client_id" and i + 1 < len(sys.argv):
            return f"Client {sys.argv[i + 1]}"
    return "Client"


def _simulation_locked_local_train(*args, **kwargs):
    """Serialize GPU training using an OS-level file lock."""
    device = kwargs.get("device", None)
    is_cuda = device is not None and getattr(device, "type", "") == "cuda"
    lock_disabled = os.environ.get("FEDMEDAI_SIMULATION_DISABLE_LOCK", "0") == "1"

    if is_cuda and not lock_disabled and FileLock is not None:
        lock_path = Path(
            os.environ.get("FEDMEDAI_SIMULATION_LOCK_PATH", "results/.simulation_gpu.lock")
        )
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        cid_str = _get_cid_str()

        t_wait = time.time()
        with FileLock(str(lock_path)):
            wait_time = time.time() - t_wait
            if wait_time > 0.5:
                print(
                    f"[{cid_str}] Acquired GPU training lock after {wait_time:.1f}s waiting.",
                    flush=True,
                )
            try:
                res = _orig_local_train(*args, **kwargs)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return res
    else:
        return _orig_local_train(*args, **kwargs)


def _simulation_locked_evaluate(self, parameters, config):
    """Serialize GPU evaluation to avoid memory spikes."""
    is_cuda = self.device is not None and getattr(self.device, "type", "") == "cuda"
    lock_disabled = os.environ.get("FEDMEDAI_SIMULATION_DISABLE_LOCK", "0") == "1"

    if is_cuda and not lock_disabled and FileLock is not None:
        lock_path = Path(
            os.environ.get("FEDMEDAI_SIMULATION_LOCK_PATH", "results/.simulation_gpu.lock")
        )
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(lock_path)):
            try:
                res = _orig_evaluate(self, parameters, config)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return res
    else:
        return _orig_evaluate(self, parameters, config)


# Patch client module hooks for simulation run
client_module._local_train = _simulation_locked_local_train
client_module.FedMedAIClient.evaluate = _simulation_locked_evaluate


def main():
    client_module.main()


if __name__ == "__main__":
    main()
