import time
import socket
import threading
from typing import Dict, Any, Optional, Callable


def ping_server(host: str, port: int, timeout: float = 2.0) -> Dict[str, Any]:
    """Test TCP socket connection latency to target server host and port."""
    start = time.perf_counter()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((host, port))
            latency_ms = (time.perf_counter() - start) * 1000.0
            return {
                "success": True,
                "latency_ms": round(latency_ms, 1),
                "message": f"Online ({round(latency_ms, 1)}ms latency)",
            }
    except socket.timeout:
        return {"success": False, "latency_ms": None, "message": "Connection Timed Out"}
    except ConnectionRefusedError:
        return {"success": False, "latency_ms": None, "message": "Connection Refused (Server not running)"}
    except Exception as e:
        return {"success": False, "latency_ms": None, "message": f"Unreachable: {e}"}


class HardwareTelemetryMonitor:
    """Background polling thread for GPU VRAM and system CPU metrics."""

    def __init__(self, interval: float = 2.0, callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.interval = interval
        self.callback = callback
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _sample(self) -> Dict[str, Any]:
        info = {
            "gpu_name": None,
            "vram_used_gb": 0.0,
            "vram_total_gb": 0.0,
            "has_gpu": False,
        }
        try:
            import torch
            if torch.cuda.is_available():
                info["has_gpu"] = True
                info["gpu_name"] = torch.cuda.get_device_name(0)
                used = torch.cuda.memory_allocated(0) / (1024 ** 3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                info["vram_used_gb"] = round(used, 2)
                info["vram_total_gb"] = round(total, 1)
        except Exception:
            pass
        return info

    def _run(self):
        while self._running:
            metrics = self._sample()
            if self.callback:
                try:
                    self.callback(metrics)
                except Exception:
                    pass
            time.sleep(self.interval)
