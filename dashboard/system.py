"""Bounded host probes. Blocking OS utilities are called through asyncio.to_thread."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from functools import lru_cache

import psutil


def lan_ips():
    found = {'127.0.0.1'}
    stats = psutil.net_if_stats()
    for name, addresses in psutil.net_if_addrs().items():
        if name in stats and stats[name].isup:
            found.update(a.address for a in addresses if a.family == socket.AF_INET)
    try:
        found.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    return sorted(found)


def endpoint(address):
    host, sep, port = address.strip().rpartition(':')
    if not sep or not host or any(c.isspace() for c in host):
        raise ValueError('Address must use IP:port or hostname:port format')
    port = int(port)
    if not 1 <= port <= 65535:
        raise ValueError('Port must be between 1 and 65535')
    return host, port


def ping(address):
    host, port = endpoint(address)
    start = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=2):
            return {'ok': True, 'ms': round((time.perf_counter() - start) * 1000, 2), 'error': None}
    except OSError as exc:
        return {'ok': False, 'ms': None, 'error': str(exc)}


def port_owners(port):
    owners = {}
    for conn in psutil.net_connections(kind='tcp'):
        if conn.status == psutil.CONN_LISTEN and conn.laddr.port == int(port) and conn.pid:
            try:
                proc = psutil.Process(conn.pid)
                owners[proc.pid] = {'pid': proc.pid, 'name': proc.name(), 'created': proc.create_time()}
            except psutil.NoSuchProcess:
                pass
    return list(owners.values())


def check_port_in_use(port):
    return bool(port_owners(port))


@lru_cache(maxsize=1)
def torch_devices():
    try:
        import torch
        if torch.cuda.is_available():
            return [(f'cuda:{i}', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
    except (ImportError, RuntimeError, OSError):
        pass
    return []


def hardware():
    vm = psutil.virtual_memory()
    data = {'cpu': psutil.cpu_percent(interval=0.1), 'ram': vm.percent,
            'ram_gb': round(vm.used / 2**30, 1), 'gpu': None, 'temperature': None,
            'vram_mb': None, 'gpu_name': 'GPU telemetry unavailable', 'devices': ['auto', 'cpu']}
    if shutil.which('nvidia-smi'):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,temperature.gpu',
                                     '--format=csv,noheader,nounits'], capture_output=True, text=True,
                                    timeout=3, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            lines = result.stdout.strip().splitlines()
            if result.returncode == 0 and lines:
                for line in lines:
                    index, name, util, memory, temp = [v.strip() for v in line.split(',')]
                    data['devices'].append(f'cuda:{int(index)}')
                _, name, util, memory, temp = [v.strip() for v in lines[0].split(',')]
                data['gpu_name'] = name
                for key, value in [('gpu', util), ('vram_mb', memory), ('temperature', temp)]:
                    try:
                        data[key] = float(value)
                    except ValueError:
                        pass
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    jetson = Path('/etc/nv_tegra_release').exists() or shutil.which('tegrastats') is not None
    if jetson:
        data['gpu_name'] = 'NVIDIA Jetson · shared memory'
        if 'cuda:0' not in data['devices']:
            data['devices'].append('cuda:0')
        for path in Path('/sys/devices').glob('platform/*gpu/load'):
            try:
                data['gpu'] = float(path.read_text()) / 10
                break
            except (OSError, ValueError):
                pass
        for zone in Path('/sys/class/thermal').glob('thermal_zone*'):
            try:
                if 'gpu' in (zone / 'type').read_text().lower():
                    data['temperature'] = float((zone / 'temp').read_text()) / 1000
                    break
            except (OSError, ValueError):
                pass
    if len(data['devices']) == 2:
        detected = torch_devices()
        data['devices'].extend(device for device, _ in detected)
        if detected:
            data['gpu_name'] = detected[0][1] + ' · utilization sensor unavailable'
    return data


def partition_data(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    clients = data.get('clients', {})
    ids = sorted(clients, key=int)
    classes = sorted({key for client in clients.values() for key in client.get('class_distribution', {})}, key=int)
    series = [{'name': f'Class {key}', 'type': 'bar', 'stack': 'train',
               'data': [clients[cid].get('class_distribution', {}).get(key, 0) for cid in ids]} for key in classes]
    if not series:
        series = [{'name': 'Samples', 'type': 'bar', 'data': [clients[cid].get('num_samples', 0) for cid in ids]}]
    return data, [f'C{cid}' for cid in ids], series
