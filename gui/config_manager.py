import os
import sys
import re
import json
import socket
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASETS = ["pathmnist", "bloodmnist", "organamnist", "dermamnist"]
STRATEGIES = [
    "fedbns",
    "fedavg",
    "fedbn",
    "scaffold",
    "fedprox",
    "scr",
    "fedsld",
    "fedsol",
    "fedswitch",
    "fednova",
]
MODELS = ["baseline_bn", "baseline", "cnn"]
LR_MODES = ["fixed", "client_loss", "server_decay"]
OPTIMIZERS = ["sgd", "adam", "adamw"]
PARTITION_MODES = ["practical", "dirichlet_0.1", "dirichlet_0.3", "pathological"]
DEVICE_TYPES = ["auto", "pc", "jetson_orin", "jetson_nano", "cpu"]


def get_default_venv_path() -> str:
    """Detect default virtual environment Python executable, strictly prioritizing project-internal .venv."""
    candidates = [
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",   # Priority 1: Inside project (Windows)
        PROJECT_ROOT / ".venv" / "bin" / "python",           # Priority 1: Inside project (Linux/Mac)
        Path(sys.executable),                                 # Priority 2: Currently running interpreter
        PROJECT_ROOT.parent / ".venv" / "Scripts" / "python.exe", # Priority 3: Fallback parent
        PROJECT_ROOT.parent / ".venv" / "bin" / "python",
    ]
    for c in candidates:
        if c.is_file():
            return str(c.resolve())
    return sys.executable


def test_venv_environment(python_exe: str) -> Dict[str, Any]:
    """Execute diagnostic checks against the target python executable."""
    code = """
import sys, json

res = {
    "python_version": sys.version.split()[0],
    "torch": False,
    "torch_version": None,
    "cuda_available": False,
    "cuda_device": None,
    "flower": False,
    "flower_version": None,
    "medmnist": False,
    "project_modules": False,
    "error": None
}

try:
    import torch
    res["torch"] = True
    res["torch_version"] = torch.__version__
    res["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        res["cuda_device"] = torch.cuda.get_device_name(0)
except Exception as e:
    res["error"] = f"PyTorch error: {e}"

try:
    import flwr
    res["flower"] = True
    res["flower_version"] = flwr.__version__
except Exception as e:
    res["error"] = f"Flower error: {e}"

try:
    import medmnist
    res["medmnist"] = True
except Exception:
    pass

try:
    import models, algorithms, client, server
    res["project_modules"] = True
except Exception:
    pass

print(json.dumps(res))
"""
    try:
        proc = subprocess.run(
            [python_exe, "-c", code],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )
        if proc.returncode == 0:
            for line in reversed(proc.stdout.splitlines()):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    return json.loads(line)
        return {
            "python_version": "Unknown",
            "torch": False,
            "flower": False,
            "error": proc.stderr.strip() or "Failed to run diagnostic probe",
        }
    except Exception as e:
        return {"python_version": "Unknown", "torch": False, "flower": False, "error": str(e)}


def get_local_ip_addresses() -> List[Tuple[str, str]]:
    """Detect local IP addresses across active network interfaces."""
    ips = []
    # Primary outbound IP
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            primary_ip = s.getsockname()[0]
            ips.append(("Primary LAN", primary_ip))
    except Exception:
        primary_ip = "127.0.0.1"

    # Enumerate local interfaces
    try:
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if ip not in [p[1] for p in ips]:
                if ip.startswith("192.168.") or ip.startswith("10.") or ip.startswith("172."):
                    ips.append(("LAN / Wi-Fi", ip))
                elif ip.startswith("100."):
                    ips.append(("Tailscale / VPN", ip))
                else:
                    ips.append(("Interface", ip))
    except Exception:
        pass

    # Fallback / localhost
    if not any(ip[1] == "127.0.0.1" for ip in ips):
        ips.append(("Loopback (Local Only)", "127.0.0.1"))

    return ips


def check_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Return True if port is currently in use / bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        res = s.connect_ex((host, port))
        return res == 0


def kill_processes_on_port(port: int) -> List[int]:
    """Find and kill processes listening on the designated port (Windows compatible)."""
    killed = []
    try:
        cmd = f"netstat -ano | findstr :{port}"
        out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            parts = line.strip().split()
            if len(parts) >= 5 and "LISTENING" in parts:
                pid = int(parts[-1])
                if pid > 0 and pid not in killed and pid != os.getpid():
                    subprocess.run(
                        f"taskkill /F /PID {pid}",
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    killed.append(pid)
    except Exception:
        pass
    return killed


def resolve_partition_file(
    dataset: str,
    partition_mode: str,
    num_clients: Optional[int] = None,
    client_id: Optional[int] = None,
    seed: int = 42,
    custom_path: Optional[str] = None,
) -> Optional[Path]:
    """Locate partition JSON matching dataset, partition configuration, and client constraints."""
    if custom_path:
        p = Path(custom_path)
        if p.is_file():
            return p

    partitions_dir = PROJECT_ROOT / "data" / "partitions"
    if not partitions_dir.is_dir():
        return None

    # 1. If explicit num_clients is provided, look for exact files with that client count
    if num_clients is not None and num_clients > 0:
        candidates = [
            partitions_dir / f"partition_{dataset}_{partition_mode}_localtest_seed{seed}_clients{num_clients}.json",
            partitions_dir / f"partition_{dataset}_{partition_mode}_seed{seed}_clients{num_clients}.json",
        ]
        for c in candidates:
            if c.is_file():
                return c

        for p in partitions_dir.glob(f"*{dataset}*{partition_mode}*clients{num_clients}*.json"):
            if p.is_file():
                return p

    # 2. If client_id is given (e.g. client 15), prioritize a partition file that contains this client ID
    if client_id is not None and client_id >= 0:
        matching_files = list(partitions_dir.glob(f"*{dataset}*{partition_mode}*.json"))
        # First check filename for clients<N> where N > client_id
        for p in matching_files:
            try:
                m = re.search(r"clients(\d+)", p.name)
                if m and int(m.group(1)) > client_id:
                    return p
            except Exception:
                pass

        # If filename didn't yield, inspect JSON content
        for p in matching_files:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                clients = meta.get("clients") or meta.get("client_data") or meta
                if isinstance(clients, dict) and str(client_id) in clients:
                    return p
            except Exception:
                pass

    # 3. Default priority pattern: clients12 then wildcard
    candidates = [
        partitions_dir / f"partition_{dataset}_{partition_mode}_localtest_seed{seed}_clients12.json",
        partitions_dir / f"partition_{dataset}_{partition_mode}_seed{seed}_clients12.json",
    ]
    for c in candidates:
        if c.is_file():
            return c

    for p in partitions_dir.glob(f"*{dataset}*{partition_mode}*.json"):
        if p.is_file():
            return p

    return None


def load_partition_distribution(
    dataset: str,
    partition_mode: str,
    client_id: int,
    num_clients: Optional[int] = None,
    seed: int = 42,
    custom_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load sample counts and label class breakdown for a specific client."""
    path = resolve_partition_file(
        dataset=dataset,
        partition_mode=partition_mode,
        num_clients=num_clients,
        client_id=client_id,
        seed=seed,
        custom_path=custom_path,
    )
    if not path or not path.is_file():
        return {
            "samples": 0,
            "distribution": {},
            "file": None,
            "num_clients": 0,
            "client_exists": False,
            "error": "No matching partition file found",
        }

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cid_str = str(client_id)
        clients = data.get("clients") or data.get("client_data") or data
        total_clients = len(clients) if isinstance(clients, dict) else int(data.get("num_clients", 0))

        if not isinstance(clients, dict) or cid_str not in clients:
            return {
                "samples": 0,
                "distribution": {},
                "file": str(path.name),
                "num_clients": total_clients,
                "client_exists": False,
                "error": f"Client {client_id} not in partition (file has {total_clients} clients: IDs 0 to {max(0, total_clients - 1)})",
            }

        client_data = clients[cid_str]
        counts = {}
        if "class_distribution" in client_data:
            counts = {str(k): int(v) for k, v in client_data["class_distribution"].items()}
        elif "class_counts" in client_data:
            counts = {str(k): int(v) for k, v in client_data["class_counts"].items()}
        elif "train_labels" in client_data:
            for l in client_data["train_labels"]:
                counts[str(l)] = counts.get(str(l), 0) + 1

        total_samples = client_data.get("num_samples") or (
            len(client_data.get("indices", [])) if "indices" in client_data else sum(counts.values())
        )
        return {
            "samples": int(total_samples),
            "distribution": counts,
            "file": str(path.name),
            "num_clients": total_clients,
            "client_exists": True,
            "error": None,
        }
    except Exception as e:
        return {
            "samples": 0,
            "distribution": {},
            "file": str(path.name) if path else None,
            "num_clients": 0,
            "client_exists": False,
            "error": str(e),
        }


def check_partition_exists(
    dataset: str,
    partition_mode: str,
    num_clients: int,
    seed: int = 42,
    custom_path: Optional[str] = None,
) -> Tuple[bool, Path]:
    """Check if the designated partition file exists, returning (exists, path)."""
    if custom_path:
        p = Path(custom_path)
        return p.is_file(), p

    partitions_dir = PROJECT_ROOT / "data" / "partitions"
    candidate = partitions_dir / f"partition_{dataset}_{partition_mode}_localtest_seed{seed}_clients{num_clients}.json"
    if candidate.is_file():
        return True, candidate

    fallback = partitions_dir / f"partition_{dataset}_{partition_mode}_seed{seed}_clients{num_clients}.json"
    if fallback.is_file():
        return True, fallback

    return False, candidate


def generate_partition_subprocess(
    dataset: str,
    partition_mode: str,
    num_clients: int,
    seed: int = 42,
    local_test: bool = True,
    python_exe: Optional[str] = None,
) -> Tuple[bool, str]:
    """Invoke experiments.run_partition in a child process to generate the target partition."""
    exe = python_exe or get_default_venv_path()
    cmd = [
        exe,
        "-m",
        "experiments.run_partition",
        "--dataset", dataset,
        "--partition_mode", partition_mode,
        "--num_clients", str(num_clients),
        "--seed", str(seed),
    ]
    if local_test:
        cmd.append("--local_test")
    else:
        cmd.append("--no-local_test")

    try:
        res = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=180,
        )
        if res.returncode == 0:
            return True, f"Successfully generated partition for {num_clients} clients ({dataset}, {partition_mode})."
        err_msg = res.stderr.strip() or res.stdout.strip()
        return False, f"Generation failed (code {res.returncode}): {err_msg[:300]}"
    except Exception as e:
        return False, f"Error generating partition: {e}"
