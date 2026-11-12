"""Durable coordinator/client supervision. A single web-service worker owns subprocesses."""
import json
import os
import shutil
import socket
import threading
import time
import uuid
from pathlib import Path
import httpx
import psutil
from portal.common import ROOT, atomic_json, connectivity, digest
from portal.processes import Process, terminate_owned
from portal.schemas import ACTIVE
from portal.storage import Store


class Busy(ValueError):
    pass


class Service:
    def __init__(self, state):
        self.state = Path(state).resolve()
        self.settings = json.loads((self.state / 'settings.json').read_text(encoding='utf-8'))
        self.mode = self.settings['mode']
        self.store = Store(self.state / 'portal.sqlite3')
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.processes = {}
        self.offsets = {}
        self.last_heartbeat = 0
        self.last_error = None

    def env(self, extra=None):
        env = dict(os.environ, PYTHONUNBUFFERED='1', MPLBACKEND='Agg', FEDMED_CPU_THREADS='1',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', FEDMED_DATA_ROOT=self.settings['data_root'])
        env.update(extra or {})
        return env

    def start(self):
        for job in self.store.all('jobs'):
            if job['status'] in ('queued', 'running'):
                if job.get('pid'):
                    terminate_owned(job['pid'], job['process_created'])
                self.store.update('jobs', job['id'], status='interrupted', error='Service restarted; retry this task.')
                (self.state / 'jobs' / job['id'] / 'request.json').unlink(missing_ok=True)
        for run in self.store.all('runs'):
            if run['status'] in ACTIVE:
                self.store.update('runs', run['id'], status='interrupted', error='Coordinator restarted. Create a new session.', ended_at=time.time())
        for part in self.store.all('participations'):
            if part.get('status') in ('training', 'ready', 'checking', 'finalizing'):
                self.store.update('participations', part['id'], status='interrupted', error='Institution service restarted. Join a new session.')
        self.task('preflight', {})
        self.thread = threading.Thread(target=self.loop, name='fedmed-supervisor', daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if hasattr(self, 'thread'):
            self.thread.join(timeout=12)
        for process in list(self.processes.values()):
            if process.poll() is None:
                process.stop()

    def busy(self):
        return any(j['status'] in ('queued', 'running') for j in self.store.all('jobs'))

    def task(self, action, payload, context=None):
        with self.lock:
            job_id = uuid.uuid4().hex
            folder = self.state / 'jobs' / job_id
            atomic_json(folder / 'request.json', dict(payload, action=action))
            job = {'id': job_id, 'action': action, 'status': 'queued', 'created_at': time.time(),
                   'context': context or {}, 'folder': str(folder)}
            self.store.put('jobs', job)
            return job

    def launch(self, job, argv, extra=None):
        folder = Path(job['folder'])
        try:
            process = Process(argv, folder / 'process.log', self.env(extra))
        except OSError as exc:
            finished = self.store.update('jobs', job['id'], status='failed', ended_at=time.time(),
                                         error=f'Could not start the configured training Python: {exc}')
            self.complete(finished)
            return False
        self.processes[job['id']] = process
        self.store.update('jobs', job['id'], status='running', started_at=time.time(), **process.identity())
        return True

    def loop(self):
        while not self.stop_event.wait(0.5):
            try:
                with self.lock:
                    self.poll_jobs()
                    if self.mode == 'coordinator':
                        self.coordinator_tick()
                if self.mode == 'institution' and time.time() - self.last_heartbeat >= 5:
                    self.last_heartbeat = time.time()
                    self.agent_tick()
            except Exception as exc:
                self.last_error = str(exc)

    def poll_jobs(self):
        jobs = self.store.all('jobs')
        for job in jobs:
            if job['status'] != 'running':
                continue
            process = self.processes.get(job['id'])
            if not process:
                continue
            self.tail_events(job)
            code = process.poll()
            timeout = 1800 if job['action'] == 'download' else 300
            if job['action'] == 'train':
                timeout = job['context']['rounds'] * job['context']['round_timeout'] * 2 + 180
            if code is None and time.time() - process.started > timeout:
                process.stop()
                code = -1
                self.store.update('jobs', job['id'], error='Task timed out.')
            if code is None:
                continue
            response_path = Path(job['folder']) / 'response.json'
            result = json.loads(response_path.read_text()) if response_path.exists() else {}
            error = result.get('error') or self.store.get('jobs', job['id']).get('error')
            okay = code == 0 and (job['action'] == 'train' or result.get('ok'))
            if not okay and not error:
                error = f"Process exited with code {code}. Open the task log for details."
                log_path = Path(job['folder']) / 'process.log'
                if log_path.exists():
                    with log_path.open('rb') as stream:
                        stream.seek(max(0, log_path.stat().st_size - 4000))
                        lines = stream.read().decode('utf-8', errors='replace').splitlines()
                    causes = [line for line in lines if 'Error:' in line or 'Exception:' in line]
                    if causes:
                        error = causes[-1][:500]
            finished = self.store.update('jobs', job['id'], status='completed' if okay else 'failed',
                                         ended_at=time.time(), result=result.get('result'), error=error)
            self.processes.pop(job['id'], None)
            (Path(job['folder']) / 'request.json').unlink(missing_ok=True)
            self.complete(finished)
        if not any(j['status'] == 'running' for j in self.store.all('jobs')):
            queued = sorted([j for j in self.store.all('jobs') if j['status'] == 'queued'], key=lambda j: j['created_at'])
            if queued:
                job = queued[0]
                folder = Path(job['folder'])
                self.launch(job, [self.settings['trainer_python'], '-u', '-m', 'portal.worker',
                                  '--request', str(folder / 'request.json'), '--response', str(folder / 'response.json')])

    def tail_events(self, job):
        path = Path(job['folder']) / 'events.jsonl'
        if not path.exists():
            return
        with path.open('r', encoding='utf-8') as stream:
            stream.seek(self.offsets.get(job['id'], 0))
            while True:
                position = stream.tell()
                line = stream.readline()
                if not line or not line.endswith('\n'):
                    self.offsets[job['id']] = position
                    break
                event = json.loads(line)
                self.store.event(event)
                if self.mode == 'coordinator' and event['client_id'] == -1:
                    run = self.store.get('runs', event['run_id'])
                    if run and run['status'] == 'running':
                        changes = {'phase': event['phase'], 'current_round': event['round']}
                        if event['phase'] == 'round_completed':
                            changes['completed_rounds'] = event['round']
                        self.store.update('runs', run['id'], **changes)
                elif self.mode == 'institution':
                    part = self.store.get('participations', event['run_id'])
                    if part and part['status'] == 'training':
                        changes = {'phase': event['phase'], 'current_round': event['round']}
                        if event['phase'] in ('evaluation_completed', 'evaluation_skipped'):
                            changes['completed_rounds'] = event['round']
                        self.store.update('participations', part['id'], **changes)

    def complete(self, job):
        ctx, result = job['context'], job.get('result')
        failed = job['status'] != 'completed'
        action = job['action']
        if action == 'preflight':
            self.store.put('settings', {'id': 'capabilities', **(result or {'ready': False, 'errors': [job['error']]})})
        elif action in ('download', 'dataset') and not failed:
            self.store.put('datasets', dict(result, id=result['dataset']))
        elif action == 'partition':
            run = self.store.get('runs', ctx['run_id'])
            if run and run['status'] == 'preparing':
                if failed or not result['capabilities']['ready']:
                    self.store.update('runs', run['id'], status='failed', error=job['error'] or 'Training environment is not ready.')
                else:
                    self.store.update('runs', run['id'], status='waiting', **result)
        elif action == 'verify':
            part = self.store.get('participations', ctx['run_id'])
            if part and part['status'] == 'checking':
                self.store.update('participations', part['id'], status='failed' if failed else 'ready',
                                  error=job['error'], partition_hash=None if failed else result['partition_hash'])
                if not failed:
                    self.store.put('datasets', dict(result['dataset'], id=result['dataset']['dataset']))
                    self.store.put('settings', dict(result['capabilities'], id='capabilities'))
                    self.store.update('participations', part['id'], train_samples=result['dataset']['train_samples'],
                                      test_samples=result['dataset']['test_samples'])
        elif action == 'model':
            if not failed:
                self.store.put('models', result)
                atomic_json(self.state / 'manifests' / (result['id'] + '.json'), result)
            if ctx.get('run_id'):
                kind = 'runs' if self.mode == 'coordinator' else 'participations'
                current = self.store.get(kind, ctx['run_id'])
                if current and current['status'] == 'finalizing':
                    self.store.update(kind, current['id'], status='failed' if failed else ('awaiting_clients' if kind == 'runs' else 'completed'),
                                      error=job['error'], model_id=None if failed else result['id'], finalized_at=time.time(),
                                      **({'ended_at': time.time()} if kind == 'participations' or failed else {}))
        elif action == 'train':
            run_id = ctx['run_id']
            kind = 'runs' if self.mode == 'coordinator' else 'participations'
            current = self.store.get(kind, run_id)
            if not current or current['status'] in ('cancelled', 'failed', 'interrupted'):
                return
            round_events = [e for e in self.store.events(run_id, limit=100000) if e['phase'] in ('round_completed', 'checkpoint_saved')]
            expected = ctx['rounds']
            if failed or not round_events or max(e['round'] for e in round_events) != expected:
                self.store.update(kind, run_id, status='failed', error=job['error'] or 'Worker ended before all rounds completed.', ended_at=time.time())
                return
            self.store.update(kind, run_id, status='finalizing')
            spec = current['spec']
            checkpoint = Path(job['folder']) / ('artifacts/final_model.pth' if self.mode == 'coordinator' else 'final_model.pth')
            if self.mode == 'coordinator' and spec['strategy'] in ('fedbn', 'fedbns'):
                self.store.update(kind, run_id, status='awaiting_clients', finalized_at=time.time())
                return
            self.task('model', {'path': str(checkpoint), 'dataset': spec['dataset'], 'model': spec['model'],
                      'strategy': spec['strategy'], 'source_run': run_id, 'source_round': expected, 'selection_rule': 'final_round',
                      'institution_id': self.settings['id'] if self.mode == 'institution' else None,
                      'evaluation': self.checkpoint_evaluation(run_id, expected)}, {'run_id': run_id})

    def checkpoint_evaluation(self, run_id, round_number):
        for event in reversed(self.store.events(run_id, limit=100000)):
            if event['round'] != round_number:
                continue
            if event['phase'] == 'round_completed' and event['client_id'] == -1:
                return {k: event['metrics'].get(k) for k in ('eval_accuracy','eval_loss','bta','bmcta')}
            if self.mode == 'institution' and event['phase'] in ('evaluation_completed','evaluation_skipped'):
                return {'eval_accuracy': event.get('accuracy'), 'eval_loss': event.get('loss'),
                        'samples': event.get('num_samples'), 'skipped': event['phase'] == 'evaluation_skipped'}
        return None

    def create_run(self, spec):
        with self.lock:
            if any(r['status'] in ACTIVE | {'finalizing', 'awaiting_clients'} for r in self.store.all('runs')):
                raise Busy('Finish or cancel the active session first.')
            for node_id in spec['node_ids']:
                node = self.store.get('nodes', node_id)
                if not node or connectivity(node.get('last_seen')) != 'online':
                    raise ValueError('Every selected institution must be online.')
            run = {'id': uuid.uuid4().hex, 'name': spec['name'], 'spec': spec, 'status': 'preparing',
                   'created_at': time.time(), 'current_round': 0, 'completed_rounds': 0,
                   'source': 'live', 'participants': {n: {'client_id': i, 'status': 'invited'} for i, n in enumerate(spec['node_ids'])}}
            self.store.put('runs', run)
            self.task('partition', {'spec': spec, 'output_dir': str(self.state / 'partitions' / run['id'])}, {'run_id': run['id']})
            return run

    def start_run(self, run_id):
        with self.lock:
            run = self.store.get('runs', run_id)
            if not run or run['status'] != 'waiting':
                raise ValueError('The session must be waiting for institutions before it can start.')
            for node_id, part in run['participants'].items():
                node = self.store.get('nodes', node_id)
                if part['status'] != 'ready' or part.get('partition_hash') != run['partition_hash'] or connectivity(node.get('last_seen')) != 'online':
                    raise ValueError('Every institution must accept and pass its readiness check.')
            if self.busy():
                raise Busy('Wait for the current preparation task to finish.')
            # Check before creating a job, so a port conflict can be retried cleanly.
            try:
                with socket.socket() as sock:
                    sock.bind(('0.0.0.0', self.settings['flower_port']))
            except OSError as exc:
                raise ValueError('The Flower port is already in use. Stop the other server or configure another port.') from exc
            spec = run['spec']
            job = self.training_job(run_id, spec)
            folder = Path(job['folder'])
            argv = [self.settings['trainer_python'], '-u', '-m', 'server.server', '--config', str(folder / 'config.yaml'),
                    '--dataset', spec['dataset'], '--model', spec['model'], '--strategy', spec['strategy'],
                    '--partition_path', run['partition_path'], '--rounds', str(spec['rounds']),
                    '--min_clients', str(len(spec['node_ids'])), '--local_epochs', str(spec['local_epochs']),
                    '--optimizer', 'sgd', '--learning_rate', str(spec['learning_rate']), '--momentum', str(spec['momentum']),
                    '--weight_decay', str(spec['weight_decay']), '--seed', str(spec['seed']), '--lr_mode', 'fixed',
                    '--local_test', '--no-server_eval', '--early_stop_patience', '0', '--scr_gamma', str(spec['scr_gamma']),
                    '--scr_head_lambda', str(spec['scr_head_lambda']), '--round_timeout', str(spec['round_timeout']),
                    '--host', '0.0.0.0', '--port', str(self.settings['flower_port']), '--save_dir', str(folder / 'artifacts')]
            if not self.launch(job, argv, self.training_env(job, run_id, -1)):
                raise ValueError(self.store.get('jobs', job['id'])['error'])
            return self.store.update('runs', run_id, status='starting', job_id=job['id'], started_at=time.time(), error=None)

    def training_job(self, run_id, spec):
        job_id = uuid.uuid4().hex
        folder = self.state / 'jobs' / job_id
        folder.mkdir(parents=True)
        (folder / 'config.yaml').write_text('{}\n')
        job = {'id': job_id, 'action': 'train', 'status': 'created', 'created_at': time.time(), 'folder': str(folder),
               'context': {'run_id': run_id, 'rounds': spec['rounds'], 'round_timeout': spec['round_timeout']}}
        self.store.put('jobs', job)
        return job

    def training_env(self, job, run_id, client_id):
        return {'FEDMED_RUN_ID': run_id, 'FEDMED_CLIENT_ID': str(client_id),
                'FEDMED_EVENT_LOG': str(Path(job['folder']) / 'events.jsonl'), 'FEDMED_OFFLINE': '1',
                'FEDMED_STRICT_ROSTER': '1', 'FEDMED_ROSTER_TIMEOUT': str(job['context']['round_timeout'])}

    def cancel(self, run_id, reason='Cancelled by operator.'):
        with self.lock:
            kind = 'runs' if self.mode == 'coordinator' else 'participations'
            current = self.store.get(kind, run_id)
            if not current:
                raise ValueError('Session not found')
            if current['status'] in ('completed', 'cancelled', 'failed', 'interrupted'):
                return current
            self.store.update(kind, run_id, status='stopping')
            for job in self.store.all('jobs'):
                if job['context'].get('run_id') == run_id and job['status'] in ('queued', 'running', 'created'):
                    process = self.processes.pop(job['id'], None)
                    if process and process.poll() is None:
                        process.stop()
                    self.store.update('jobs', job['id'], status='cancelled', error=reason, ended_at=time.time())
                    (Path(job['folder']) / 'request.json').unlink(missing_ok=True)
            return self.store.update(kind, run_id, status='cancelled', error=reason, ended_at=time.time())

    def coordinator_tick(self):
        for run in self.store.all('runs'):
            if run['status'] == 'starting':
                try:
                    with socket.create_connection(('127.0.0.1', self.settings['flower_port']), timeout=0.15):
                        self.store.update('runs', run['id'], status='running', phase='waiting_for_clients')
                except OSError:
                    if time.time() - run['started_at'] > 120:
                        self.cancel(run['id'], 'Flower server did not become ready within 120 seconds.')
                        self.store.update('runs', run['id'], status='failed')
            if run['status'] in ('running', 'awaiting_clients'):
                statuses = [p['status'] for p in run['participants'].values()]
                if any(s in ('failed', 'interrupted', 'cancelled') for s in statuses):
                    self.cancel(run['id'], 'A selected institution failed or stopped. Inspect its status and create a new run.')
                    self.store.update('runs', run['id'], status='failed')
                elif run['status'] == 'awaiting_clients':
                    if all(s == 'completed' for s in statuses):
                        self.store.update('runs', run['id'], status='completed', ended_at=time.time())
                        self.save_experiment(self.store.get('runs', run['id']))
                    elif time.time() - run.get('finalized_at', time.time()) > 180:
                        self.store.update('runs', run['id'], status='failed', error='Institution checkpoint finalization timed out.', ended_at=time.time())

    def connection(self):
        return self.store.get('settings', 'connection')

    def save_experiment(self, run):
        spec = run['spec']
        rounds = {e['round']: e['metrics'] for e in self.store.events(run['id'], limit=100000)
                  if e['phase'] == 'round_completed' and e['client_id'] == -1}
        curves = [rounds[i] for i in sorted(rounds)]
        last = curves[-1] if curves else {}
        self.store.put('experiments', {'id': run['id'], 'name': run['name'], 'status': run['status'],
            'dataset': spec['dataset'], 'strategy': spec['strategy'], 'model': spec['model'],
            'seed': str(spec['seed']), 'partition': spec['partition'], 'partition_hash': run['partition_hash'],
            'num_clients': len(spec['node_ids']), 'source': 'live', 'rounds': len(curves),
            'bta': last.get('bta'), 'bmcta': last.get('bmcta'), 'final_accuracy': last.get('eval_accuracy'),
            'loss': last.get('eval_loss'),
            'time_seconds': run['ended_at']-run['started_at'], 'curves': curves,
            'configuration': {k: spec[k] for k in ('local_epochs','batch_size','learning_rate','momentum','weight_decay')}})

    def request(self, method, path, **kwargs):
        conn = self.connection()
        if not conn:
            raise ValueError('Connect this institution to a coordinator first.')
        with httpx.Client(timeout=12, trust_env=False) as client:
            response = client.request(method, conn['url'] + '/api/v1/agents' + path,
                                      headers={'Authorization': 'Bearer ' + conn['token']}, **kwargs)
            response.raise_for_status()
            return response

    def agent_tick(self):
        conn = self.connection()
        if not conn:
            return
        pending = self.store.events(after=conn.get('event_cursor', 0), limit=500)
        parts = [{'id': p['id'], 'status': p['status'], 'error': p.get('error'), 'partition_hash': p.get('partition_hash'),
                  'train_samples': p.get('train_samples'), 'test_samples': p.get('test_samples')}
                 for p in self.store.all('participations')[:20]]
        public_models = [{k: v for k, v in m.items() if k != 'path'} for m in self.store.all('models')[:100]]
        payload = {'capabilities': self.store.get('settings', 'capabilities', {}), 'telemetry': self.telemetry(),
                   'participations': parts, 'events': pending, 'models': public_models}
        try:
            answer = self.request('POST', '/heartbeat', json=payload).json()['data']
            with self.lock:
                self.store.update('settings', 'connection', last_seen=time.time(), error=None,
                                  event_cursor=pending[-1]['seq'] if pending else conn.get('event_cursor', 0))
                for run in answer['runs']:
                    part = self.store.get('participations', run['id'])
                    if part is None:
                        part = dict(run, status='invited', coordinator_status=run['status'])
                    else:
                        part.update(coordinator_status=run['status'])
                    self.store.put('participations', part)
                    if run['status'] in ('cancelled', 'failed', 'interrupted', 'stopping') and part['status'] in ('checking', 'ready', 'training', 'finalizing', 'invited'):
                        self.cancel(part['id'], 'Coordinator ended this session.')
                    elif run['status'] == 'running' and part['status'] == 'ready':
                        self.start_client(part)
                self.store.put('settings', {'id': 'remote_models', 'models': answer['models']})
        except Exception as exc:
            self.store.update('settings', 'connection', error=str(exc))

    def accept(self, run_id):
        with self.lock:
            part = self.store.get('participations', run_id)
            if not part or part['status'] not in ('invited', 'failed') or part['coordinator_status'] != 'waiting':
                raise ValueError('This invitation cannot be accepted now.')
            if self.busy():
                raise Busy('Wait for the current device task to finish.')
            blob = self.request('GET', f'/runs/{run_id}/partition').content
            path = self.state / 'partitions' / f'{run_id}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
            if digest(path) != part['partition_hash']:
                raise ValueError('Partition checksum mismatch')
            self.store.update('participations', run_id, status='checking', partition_path=str(path), error=None)
            return self.task('verify', {'dataset': part['spec']['dataset'], 'partition_path': str(path),
                  'client_id': part['client_id'], 'partition_hash': part['partition_hash'], 'dataset_hash': part['dataset_hash'],
                  'expected_code_hash': part['capabilities']['code_hash'], 'expected_flower': part['capabilities']['flwr']}, {'run_id': run_id})

    def start_client(self, part):
        if self.busy():
            return
        spec = part['spec']
        job = self.training_job(part['id'], spec)
        folder = Path(job['folder'])
        argv = [self.settings['trainer_python'], '-u', '-m', 'client.client', '--config', str(folder / 'config.yaml'),
                '--client_id', str(part['client_id']), '--server_address', part['flower_address'],
                '--dataset', spec['dataset'], '--partition_path', part['partition_path'], '--num_clients', str(len(spec['node_ids'])),
                '--model', spec['model'], '--strategy', spec['strategy'], '--local_epochs', str(spec['local_epochs']),
                '--batch_size', str(spec['batch_size']), '--learning_rate', str(spec['learning_rate']), '--optimizer', 'sgd',
                '--momentum', str(spec['momentum']), '--weight_decay', str(spec['weight_decay']), '--lr_mode', 'fixed',
                '--seed', str(spec['seed']), '--device_type', self.settings['device_type'], '--local_test', '--no-augment',
                '--no-random_crop', '--no-random_flip', '--no-pin_memory', '--num_workers', '0', '--client_dir', str(folder / 'metrics')]
        extra = self.training_env(job, part['id'], part['client_id'])
        extra['FEDMED_LOCAL_CHECKPOINT'] = str(folder / 'final_model.pth')
        if self.launch(job, argv, extra):
            self.store.update('participations', part['id'], status='training', job_id=job['id'], started_at=time.time())

    def telemetry(self):
        memory = psutil.virtual_memory()
        return {'cpu_percent': psutil.cpu_percent(), 'ram_percent': memory.percent,
                'ram_used_mb': round(memory.used/1024**2), 'ram_total_mb': round(memory.total/1024**2),
                'temperature': None, 'gpu_memory_mb': None}

    def fetch_model(self, model_id):
        models = self.store.get('settings', 'remote_models', {}).get('models', [])
        manifest = next((m for m in models if m['id'] == model_id and m['scope'] == 'global'), None)
        if not manifest:
            raise ValueError('Global model not available')
        if self.busy():
            raise Busy('Wait for this device to finish its current task.')
        blob = self.request('GET', f'/models/{model_id}/file').content
        if len(blob) > 50_000_000:
            raise ValueError('Checkpoint exceeds the 50 MB limit')
        path = self.state / 'models' / f'{model_id}.pth'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        if digest(path) != manifest['checksum']:
            path.unlink()
            raise ValueError('Model checksum mismatch')
        return self.task('model', {k: manifest[k] for k in ('dataset','model','strategy','selection_rule','source_run','source_round')} |
                         {'path': str(path), 'evaluation': manifest.get('evaluation')})
