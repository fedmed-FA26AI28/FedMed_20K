import os
import sys
import time
import socket
import ctypes
import threading
from typing import Dict, Any, Optional, Callable

# Try importing psutil for CPU and RAM telemetry
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


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


class _NVMLBackend:
    """Ultra-low-overhead direct NVML ctypes binding for NVIDIA GPU & VRAM telemetry."""

    def __init__(self):
        self.available = False
        self.handle = None
        self.gpu_name = "NVIDIA GPU"
        self._init_nvml()

    def _init_nvml(self):
        try:
            dll_name = "nvml.dll" if sys.platform.startswith("win") else "libnvidia-ml.so.1"
            self.nvml = ctypes.CDLL(dll_name)
            # nvmlInit_v2 returns 0 on NVML_SUCCESS
            if self.nvml.nvmlInit_v2() == 0:
                self.handle = ctypes.c_void_p()
                self.nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle))
                name_buf = ctypes.create_string_buffer(96)
                self.nvml.nvmlDeviceGetName(self.handle, name_buf, 96)
                raw_name = name_buf.value.decode("utf-8", errors="ignore")
                cleaned = raw_name.replace("NVIDIA", "").replace("GeForce", "").strip()
                self.gpu_name = cleaned or raw_name
                self.available = True
        except Exception:
            self.available = False

    def query(self) -> Optional[Dict[str, Any]]:
        if not self.available or not self.handle:
            return None
        try:
            class nvmlUtilization_t(ctypes.Structure):
                _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

            class nvmlMemory_t(ctypes.Structure):
                _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

            mem = nvmlMemory_t()
            util = nvmlUtilization_t()
            self.nvml.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(mem))
            self.nvml.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(util))

            temp = ctypes.c_uint(0)
            try:
                self.nvml.nvmlDeviceGetTemperature(self.handle, 0, ctypes.byref(temp))
                gpu_temp = int(temp.value)
            except Exception:
                gpu_temp = None

            used_gb = round(mem.used / (1024 ** 3), 2)
            total_gb = round(mem.total / (1024 ** 3), 2)
            vram_pct = round((mem.used / mem.total) * 100, 1) if mem.total > 0 else 0.0

            return {
                "has_gpu": True,
                "gpu_name": self.gpu_name,
                "gpu_util": float(util.gpu),
                "gpu_temp": gpu_temp,
                "vram_used_gb": used_gb,
                "vram_total_gb": total_gb,
                "vram_percent": vram_pct,
            }
        except Exception:
            return None

    def shutdown(self):
        if self.available:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass
            self.available = False


class HardwareTelemetryMonitor:
    """Real-time background hardware telemetry monitor for CPU, RAM, and GPU/VRAM."""

    def __init__(self, interval: float = 1.2, callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.interval = interval
        self.callback = callback
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._nvml = _NVMLBackend()

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._nvml:
            self._nvml.shutdown()

    def _sample_cpu_ram(self) -> Dict[str, Any]:
        """Fetch system CPU and RAM usage with native OS fallbacks."""
        cpu_pct = 0.0
        ram_used = 0.0
        ram_total = 0.0
        ram_pct = 0.0

        if HAS_PSUTIL:
            try:
                cpu_pct = round(psutil.cpu_percent(interval=None), 1)
                vm = psutil.virtual_memory()
                ram_used = round(vm.used / (1024 ** 3), 2)
                ram_total = round(vm.total / (1024 ** 3), 2)
                ram_pct = round(vm.percent, 1)
                return {
                    "cpu_percent": cpu_pct,
                    "ram_used_gb": ram_used,
                    "ram_total_gb": ram_total,
                    "ram_percent": ram_pct,
                }
            except Exception:
                pass

        # Windows Native ctypes fallback if psutil is unavailable or fails
        if sys.platform.startswith("win"):
            try:
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                ram_pct = float(stat.dwMemoryLoad)
                ram_total = round(stat.ullTotalPhys / (1024 ** 3), 2)
                ram_used = round((stat.ullTotalPhys - stat.ullAvailPhys) / (1024 ** 3), 2)
            except Exception:
                pass

        return {
            "cpu_percent": cpu_pct,
            "ram_used_gb": ram_used,
            "ram_total_gb": ram_total,
            "ram_percent": ram_pct,
        }

    def _sample_gpu(self) -> Dict[str, Any]:
        """Fetch GPU and VRAM metrics with NVML -> nvidia-smi -> PyTorch -> CPU fallbacks."""
        # 1. Native NVML query
        res = self._nvml.query()
        if res:
            return res

        # 2. nvidia-smi subprocess query
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"],
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
                text=True,
                timeout=0.6,
            ).strip()
            parts = [p.strip() for p in out.split(",")]
            if len(parts) >= 4:
                raw_name = parts[0]
                cleaned = raw_name.replace("NVIDIA", "").replace("GeForce", "").strip()
                gpu_util = float(parts[1])
                used_mb = float(parts[2])
                total_mb = float(parts[3])
                temp = int(float(parts[4])) if len(parts) >= 5 else None
                return {
                    "has_gpu": True,
                    "gpu_name": cleaned or raw_name,
                    "gpu_util": gpu_util,
                    "gpu_temp": temp,
                    "vram_used_gb": round(used_mb / 1024, 2),
                    "vram_total_gb": round(total_mb / 1024, 2),
                    "vram_percent": round((used_mb / total_mb) * 100, 1) if total_mb > 0 else 0.0,
                }
        except Exception:
            pass

        # 3. PyTorch fallback
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                cleaned = name.replace("NVIDIA", "").replace("GeForce", "").strip()
                used = torch.cuda.memory_allocated(0) / (1024 ** 3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                pct = round((used / total) * 100, 1) if total > 0 else 0.0
                return {
                    "has_gpu": True,
                    "gpu_name": cleaned or name,
                    "gpu_util": 0.0,
                    "gpu_temp": None,
                    "vram_used_gb": round(used, 2),
                    "vram_total_gb": round(total, 1),
                    "vram_percent": pct,
                }
        except Exception:
            pass

        return {
            "has_gpu": False,
            "gpu_name": "CPU Mode",
            "gpu_util": 0.0,
            "gpu_temp": None,
            "vram_used_gb": 0.0,
            "vram_total_gb": 0.0,
            "vram_percent": 0.0,
        }

    def _sample(self) -> Dict[str, Any]:
        metrics = self._sample_cpu_ram()
        gpu_metrics = self._sample_gpu()
        metrics.update(gpu_metrics)
        return metrics

    def _run(self):
        # Warmup CPU measurement so first sample isn't 0.0
        if HAS_PSUTIL:
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                pass

        while self._running:
            metrics = self._sample()
            if self.callback:
                try:
                    self.callback(metrics)
                except Exception:
                    pass
            time.sleep(self.interval)
