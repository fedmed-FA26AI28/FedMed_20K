"""Async process lifecycle for the FL orchestration service."""
from __future__ import annotations
import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from uuid import uuid4
import psutil
import yaml
from dashboard.metrics import ANSI, MetricStream, log_level
from dashboard.system import endpoint, check_port_in_use, hardware

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'results' / 'dashboard'
STRATEGIES = ['fedavg', 'fedprox', 'fednova', 'scaffold', 'fedbn', 'fedbns', 'scr', 'fedsld', 'fedsol', 'fedswitch']
DATASETS = ['pathmnist', 'bloodmnist', 'organamnist']
PARTITIONS = ['practical', 'pathological', 'dirichlet_0.1', 'dirichlet_0.3', 'dirichlet_1.0']
MODES = {'server': 'FL Server', 'client': 'Local Client', 'simulation': 'Simulation'}
ACTIVE = ('starting', 'running', 'stopping')


def defaults(mode):
    name = 'simulation.yaml' if mode == 'simulation' else 'experiment.yaml'
    return yaml.safe_load((ROOT / 'configs' / name).read_text(encoding='utf-8'))


def prepare_config(mode, values, web_port=None):
    if mode not in MODES:
        raise ValueError('Invalid execution mode')
    config = defaults(mode)
    config.update(device='auto', client_id=0, server_port=8081)
    config.update(values)
    for key, options in [('dataset', DATASETS), ('strategy', STRATEGIES), ('partition_mode', PARTITIONS),
                         ('model', ['cnn', 'baseline', 'baseline_bn']), ('optimizer', ['sgd', 'adam', 'adamw'])]:
        if config.get(key) not in options:
            raise ValueError(f'{key}: invalid value')
    for key, low, high in [('rounds', 1, 100), ('num_clients', 1, 1000), ('local_epochs', 1, 10000),
                          ('batch_size', 1, 100000), ('seed', 0, 2**32-1), ('client_id', 0, 999), ('server_port', 1, 65535)]:
        value = float(config[key])
        if not math.isfinite(value) or int(value) != value or not low <= value <= high:
            raise ValueError(f'{key} must be an integer between {low}–{high}')
        config[key] = int(value)
    for key in ['min_fit_clients', 'min_available_clients']:
        value = float(values.get(key, config['num_clients']))
        if not math.isfinite(value) or int(value) != value:
            raise ValueError(f'{key} must be an integer')
        config[key] = int(value)
    if not 1 <= config['min_fit_clients'] <= config['min_available_clients'] <= config['num_clients']:
        raise ValueError('Require 1 ≤ Min Fit ≤ Min Available ≤ Total clients')
    for key in ['learning_rate', 'fraction_fit', 'proximal_mu']:
        value = float(config.get(key, {'fraction_fit': 1, 'proximal_mu': 0.01}.get(key, 0.01)))
        if not math.isfinite(value) or value < 0 or (key != 'proximal_mu' and value == 0):
            raise ValueError(f'{key}: invalid value')
        config[key] = value
    if config['fraction_fit'] > 1:
        raise ValueError('Fraction Fit must be in (0, 1]')
    if config['client_id'] >= config['num_clients']:
        raise ValueError('Client ID must be less than the total client count')
    if not re.fullmatch(r'auto|cpu|cuda(?::\d+)?', str(config['device'])):
        raise ValueError('Invalid compute device')
    host = str(config['server_host']).strip()
    endpoint(f'{host}:{config["server_port"]}')
    if mode == 'server' and config['server_port'] == web_port:
        raise ValueError('The gRPC port conflicts with the web port. Choose a different gRPC port.')
    target = values.get('server_address', f'{host}:{config["server_port"]}')
    endpoint(target)
    config.update(fl_rounds=config['rounds'], min_clients=config['min_available_clients'],
                  server_address=target, server_host=host, name=None)
    if mode == 'simulation':
        config.update(datasets=[config['dataset']], strategies=[config['strategy']], partitions=[config['partition_mode']])
        config.update(min_fit_clients=config['num_clients'], min_available_clients=config['num_clients'],
                      min_clients=config['num_clients'], fraction_fit=1.0)
    return config


def command(mode, config_path, output, config):
    module = {'simulation': 'experiments.run_simulation', 'server': 'server.server', 'client': 'client.client'}[mode]
    cmd = [sys.executable, '-u', '-m', module, '--config', str(config_path)]
    if mode == 'client':
        cmd += ['--client_id', str(config['client_id']), '--server_address', config['server_address']]
    else:
        cmd += ['--output_dir', str(output)]
    return cmd


@dataclass
class Job:
    id: str
    mode: str
    config: dict
    folder: Path
    state: str = 'starting'
    process: asyncio.subprocess.Process | None = None
    task: asyncio.Task | None = None
    metrics: MetricStream = field(default_factory=MetricStream)
    started: float = field(default_factory=time.time)
    ended: float | None = None
    exit_code: int | None = None


class Controller:
    def __init__(self, runs=RUNS, web_port=8080):
        self.runs, self.web_port = runs, web_port
        self.jobs = {}
        self.logs = deque(maxlen=4000)
        self.sequence = 0
        self.lock = asyncio.Lock()
        self.telemetry = {}
        self.telemetry_task = None

    def log(self, job, text):
        text = ANSI.sub('', text).rstrip()
        self.sequence += 1
        self.logs.append({'seq': self.sequence, 'job': job.id, 'time': time.strftime('%H:%M:%S'),
                          'level': log_level(text), 'text': text})
        job.metrics.feed(text)

    async def start(self, mode, values):
        config = prepare_config(mode, values, self.web_port)
        async with self.lock:
            active = [j for j in self.jobs.values() if j.state in ACTIVE]
            if any(j.mode == 'simulation' for j in active) or (mode == 'simulation' and active):
                raise ValueError('Simulation must run separately from other server and client jobs.')
            if mode == 'server' and any(j.mode == 'server' for j in active):
                raise ValueError('An FL server is already running.')
            if mode == 'client' and any(j.mode == 'client' and j.config['client_id'] == config['client_id']
                                      and j.config['server_address'] == config['server_address'] for j in active):
                raise ValueError('This client ID is already running against the same server.')
            if mode == 'server' and await asyncio.to_thread(check_port_in_use, config['server_port']):
                raise ValueError('The gRPC port is busy. Choose a different port or stop its owner before starting.')
            job_id = f'{mode}-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:6]}'
            folder = self.runs / job_id
            await asyncio.to_thread(folder.mkdir, parents=True)
            path = folder / 'config.yaml'
            await asyncio.to_thread(path.write_text, yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
            job = Job(job_id, mode, config, folder)
            job.metrics.client = mode == 'client'
            job.metrics.total = config['rounds']
            self.jobs[job_id] = job
            env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', FEDMED_DASHBOARD_STREAM='1')
            device = config['device']
            if device == 'cpu':
                env['CUDA_VISIBLE_DEVICES'] = ''
            elif device.startswith('cuda'):
                env['CUDA_VISIBLE_DEVICES'] = device.split(':')[-1] if ':' in device else '0'
            cmd = command(mode, path, folder / 'artifacts', config)
            try:
                kwargs = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
                job.process = await asyncio.create_subprocess_exec(*cmd, cwd=ROOT, env=env,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, limit=8*1024*1024, **kwargs)
                job.state = 'running'
                self.log(job, '[INFO] ' + subprocess.list2cmdline(cmd))
                job.task = asyncio.create_task(self._stream(job))
            except OSError as exc:
                job.state, job.ended = 'failed', time.time()
                self.log(job, f'[ERROR] {exc}')
                raise
            return job

    async def _stream(self, job):
        import aiofiles
        try:
            async with aiofiles.open(job.folder / 'console.log', 'w', encoding='utf-8') as output:
                while raw := await job.process.stdout.readline():
                    line = raw.decode('utf-8', errors='replace')
                    await output.write(line)
                    await output.flush()
                    self.log(job, line)
            job.exit_code = await job.process.wait()
            job.state = 'stopped' if job.state == 'stopping' else ('completed' if job.exit_code == 0 else 'failed')
        except Exception as exc:
            self.log(job, f'[ERROR] Stream: {exc}')
            await self.stop(job.id)
            job.state = 'failed'
        finally:
            job.ended = time.time()
            self.log(job, f'[INFO] {job.state}; exit code={job.exit_code}')
            await asyncio.to_thread((job.folder / 'summary.json').write_text,
                                    json.dumps(self.summary(job), ensure_ascii=False, indent=2), encoding='utf-8')

    async def stop(self, job_id):
        job = self.jobs[job_id]
        if job.state not in ('running', 'starting') or job.process is None:
            return
        job.state = 'stopping'
        proc = job.process
        try:
            parent = psutil.Process(proc.pid)
            children = await asyncio.to_thread(parent.children, recursive=True)
        except psutil.NoSuchProcess:
            children = []
        try:
            if proc.returncode is None:
                if os.name == 'nt':
                    try:
                        proc.send_signal(signal.CTRL_BREAK_EVENT)
                    except (OSError, ValueError):
                        self.log(job, '[WARNING] Could not send CTRL_BREAK; taskkill will be used after the timeout.')
                else:
                    os.killpg(proc.pid, signal.SIGINT)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=6)
                except asyncio.TimeoutError:
                    if os.name == 'nt':
                        killer = await asyncio.create_subprocess_exec('taskkill', '/T', '/F', '/PID', str(proc.pid),
                            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                        await killer.wait()
                    else:
                        os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=3)
                    except asyncio.TimeoutError:
                        if os.name != 'nt':
                            os.killpg(proc.pid, signal.SIGKILL)
                        else:
                            proc.kill()
                        await proc.wait()
        except ProcessLookupError:
            pass
        finally:
            for child in children:
                try:
                    await asyncio.to_thread(child.kill)
                except psutil.NoSuchProcess:
                    pass
            if children:
                await asyncio.to_thread(psutil.wait_procs, children, timeout=3)

    def summary(self, job):
        return {'id': job.id, 'mode': job.mode, 'state': job.state, 'exit_code': job.exit_code,
                'started': job.started, 'ended': job.ended, 'config': job.config,
                'rounds': job.metrics.records(), 'nodes': list(job.metrics.nodes.values()),
                'client_histories': {cid: job.metrics.client_records(cid) for cid in job.metrics.node_history}}

    async def monitor(self):
        while True:
            try:
                self.telemetry = await asyncio.to_thread(hardware)
            except Exception as exc:
                self.telemetry = {'error': str(exc)}
            await asyncio.sleep(3)

    async def startup(self):
        self.telemetry_task = asyncio.create_task(self.monitor())

    async def shutdown(self):
        if self.telemetry_task:
            self.telemetry_task.cancel()
        await asyncio.gather(*(self.stop(job.id) for job in self.jobs.values()), return_exceptions=True)
        await asyncio.gather(*(j.task for j in self.jobs.values() if j.task), return_exceptions=True)

