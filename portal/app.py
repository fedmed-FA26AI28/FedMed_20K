import base64
import json
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse
import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from portal.common import ROOT, connectivity, contained_path, password_matches, token_hash
from portal.history import scan
from portal.schemas import (ACTIVE, ConnectRequest, DatasetRequest, Envelope, Heartbeat,
                            ImportModel, Login, PairRequest, PredictRequest, RunSpec)
from portal.service import Busy, Service


def create_app(state, start_service=True):
    service = Service(state)
    store, settings = service.store, service.settings
    cookie = 'fedmed_' + settings['id'][:12]

    @asynccontextmanager
    async def lifespan(app):
        if start_service:
            service.start()
        yield
        if start_service:
            service.close()

    app = FastAPI(title='FedMedAI Portal', version='1.0.0', lifespan=lifespan)
    app.state.service = service

    @app.middleware('http')
    async def boundaries(request, call_next):
        if request.method not in ('GET','HEAD','OPTIONS') and not request.url.path.startswith('/api/v1/agents/'):
            origin = request.headers.get('origin')
            if request.headers.get('x-fedmed-request') != '1' or (origin and urlparse(origin).netloc != request.headers.get('host')):
                return JSONResponse({'detail': 'Request origin could not be verified.'}, status_code=403)
        if int(request.headers.get('content-length', '0') or 0) > 8_000_000:
            return JSONResponse({'detail': 'Request is too large.'}, status_code=413)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'same-origin'
        response.headers['X-Frame-Options'] = 'DENY'
        if request.url.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({'detail': str(exc)}, status_code=409 if isinstance(exc, Busy) else 400)

    def operator(request: Request):
        token = request.cookies.get(cookie)
        session = store.get('sessions', token_hash(token)) if token else None
        if not session or session['expires'] < time.time():
            raise HTTPException(401, 'Sign in to continue.')
        return session

    def coordinator(session=Depends(operator)):
        if service.mode != 'coordinator':
            raise HTTPException(403, 'This action belongs to the coordinator.')
        return session

    def institution(session=Depends(operator)):
        if service.mode != 'institution':
            raise HTTPException(403, 'This action belongs to an institution.')
        return session

    def agent(request: Request):
        if service.mode != 'coordinator':
            raise HTTPException(403, 'Coordinator endpoint only.')
        authorization = request.headers.get('authorization', '')
        if not authorization.startswith('Bearer '):
            raise HTTPException(401, 'Device authentication required.')
        hashed = token_hash(authorization[7:])
        node = next((n for n in store.all('nodes') if secrets.compare_digest(n['token_hash'], hashed)), None)
        if node is None:
            raise HTTPException(401, 'Unknown device token.')
        return node

    def envelope(value):
        return {'data': value}

    def clean_job(job):
        return {k: v for k, v in job.items() if k not in ('folder','pid','process_created')}

    def clean_model(model):
        return {k: v for k, v in model.items() if k != 'path'}

    def node_view(node):
        return {k: v for k, v in node.items() if k != 'token_hash'} | {'connection_status': connectivity(node.get('last_seen'))}

    @app.get('/api/v1/status', response_model=Envelope)
    def status():
        return envelope({'mode': service.mode, 'name': settings['name'], 'version': '1.0.0'})

    @app.post('/api/v1/session', response_model=Envelope)
    def login(body: Login, request: Request, response: Response):
        key = token_hash((request.client.host if request.client else '') + body.username)
        attempts = store.get('attempts', key, {'id': key, 'count': 0, 'since': time.time()})
        if time.time() - attempts['since'] > 600:
            attempts.update(count=0, since=time.time())
        if attempts['count'] >= 10:
            raise HTTPException(429, 'Too many sign-in attempts. Try again in ten minutes.')
        if body.username != settings['username'] or not password_matches(body.password, settings['password_hash']):
            attempts['count'] += 1
            store.put('attempts', attempts)
            raise HTTPException(401, 'Incorrect username or password.')
        store.delete('attempts', key)
        token = secrets.token_urlsafe(32)
        store.put('sessions', {'id': token_hash(token), 'username': body.username, 'expires': time.time()+12*3600})
        response.set_cookie(cookie, token, httponly=True, samesite='strict', max_age=12*3600,
                            secure=request.url.scheme == 'https')
        return envelope({'username': body.username, 'mode': service.mode, 'name': settings['name']})

    @app.get('/api/v1/session', response_model=Envelope)
    def session_info(session=Depends(operator)):
        return envelope({'username': session['username'], 'mode': service.mode, 'name': settings['name']})

    @app.delete('/api/v1/session', response_model=Envelope)
    def logout(request: Request, response: Response, session=Depends(operator)):
        store.delete('sessions', session['id'])
        response.delete_cookie(cookie)
        return envelope(True)

    @app.get('/api/v1/overview', response_model=Envelope)
    def overview(session=Depends(operator)):
        conn = service.connection()
        return envelope({'mode': service.mode, 'name': settings['name'], 'instance_id': settings['id'],
                         'capabilities': store.get('settings', 'capabilities'), 'telemetry': service.telemetry(),
                         'connection': {k: v for k, v in conn.items() if k not in ('token','event_cursor')} if conn else None,
                         'nodes': [node_view(n) for n in store.all('nodes')],
                         'runs': store.all('runs') if service.mode == 'coordinator' else store.all('participations'),
                         'jobs': [clean_job(j) for j in store.all('jobs')[:25]],
                         'models': [clean_model(m) for m in store.all('models')],
                         'datasets': store.all('datasets'), 'service_error': service.last_error})

    @app.post('/api/v1/preflight', response_model=Envelope)
    def preflight(session=Depends(operator)):
        if service.busy():
            raise Busy('A task is already running. Wait before checking the environment.')
        return envelope(clean_job(service.task('preflight', {})))

    @app.get('/api/v1/jobs/{job_id}', response_model=Envelope)
    def job_detail(job_id: str, session=Depends(operator)):
        job = store.get('jobs', job_id)
        if not job:
            raise HTTPException(404, 'Task not found')
        return envelope(clean_job(job))

    @app.get('/api/v1/jobs/{job_id}/log', response_model=Envelope)
    def job_log(job_id: str, session=Depends(operator)):
        job = store.get('jobs', job_id)
        if not job:
            raise HTTPException(404, 'Task not found')
        path = Path(job['folder']) / 'process.log'
        if not path.exists():
            return envelope('Task has not started yet.')
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 32000))
            return envelope(stream.read().decode('utf-8', errors='replace'))

    @app.post('/api/v1/pairing-tokens', response_model=Envelope)
    def pair_token(session=Depends(coordinator)):
        value = secrets.token_urlsafe(24)
        store.put('pairing', {'id': token_hash(value), 'expires': time.time()+900})
        return envelope({'token': value, 'expires_in': 900,
                         'coordinator_url': f"http://{settings['advertise_host']}:{settings['port']}"})

    @app.post('/api/v1/agents/pair', response_model=Envelope)
    def pair(body: PairRequest):
        if service.mode != 'coordinator':
            raise HTTPException(403, 'Coordinator endpoint only')
        with service.lock:
            saved = store.get('pairing', token_hash(body.token))
            if not saved or saved['expires'] < time.time():
                raise HTTPException(403, 'Pairing token is invalid or expired.')
            if any(body.instance_id in r.get('participants', {}) and r['status'] in ACTIVE for r in store.all('runs')):
                raise HTTPException(409, 'Device is in an active session. Cancel that session before pairing again.')
            value = secrets.token_urlsafe(32)
            store.delete('pairing', saved['id'])
            store.put('nodes', {'id': body.instance_id, 'name': body.name, 'device_type': body.device_type,
                               'token_hash': token_hash(value), 'last_seen': time.time(), 'capabilities': {}, 'telemetry': {}})
        return envelope({'node_id': body.instance_id, 'token': value})

    @app.post('/api/v1/connection', response_model=Envelope)
    def connect(body: ConnectRequest, session=Depends(institution)):
        parsed = urlparse(body.coordinator_url)
        if parsed.scheme not in ('http','https') or not parsed.hostname or parsed.username or parsed.password or parsed.path not in ('','/'):
            raise ValueError('Use the coordinator URL, for example http://192.168.1.10:8000.')
        if any(p['status'] in ('training','ready','checking','finalizing') for p in store.all('participations')):
            raise Busy('Finish the current participation before reconnecting.')
        url = body.coordinator_url.rstrip('/')
        try:
            with httpx.Client(timeout=10, trust_env=False) as client:
                response = client.post(url+'/api/v1/agents/pair', json={'token': body.token, 'name': settings['name'],
                               'instance_id': settings['id'], 'device_type': settings['device_type']})
                response.raise_for_status()
                data = response.json()['data']
        except httpx.HTTPError as exc:
            raise ValueError(f'Could not pair with coordinator: {exc}') from exc
        store.put('settings', {'id': 'connection', 'url': url, 'node_id': data['node_id'], 'token': data['token'], 'last_seen': 0})
        return envelope({'connected': True})

    @app.post('/api/v1/agents/heartbeat', response_model=Envelope)
    def heartbeat(body: Heartbeat, node=Depends(agent)):
        with service.lock:
            store.update('nodes', node['id'], capabilities=body.capabilities, telemetry=body.telemetry,
                         last_seen=time.time(), models=[m for m in body.models if m.get('scope') == 'institution' and m.get('institution_id') == node['id']])
            for part in body.participations:
                run = store.get('runs', part.get('id'))
                if run and node['id'] in run['participants']:
                    allowed = {'invited','checking','ready','training','finalizing','completed','failed','interrupted','cancelled'}
                    if part.get('status') not in allowed:
                        continue
                    run['participants'][node['id']].update({k: part.get(k) for k in ('status','error','partition_hash','train_samples','test_samples')})
                    store.put('runs', run)
            for event in body.events:
                run = store.get('runs', event.get('run_id'))
                if run and node['id'] in run['participants'] and event.get('client_id') == run['participants'][node['id']]['client_id']:
                    if all(k in event for k in ('id','time','round','phase')):
                        store.event({k: v for k, v in event.items() if k != 'seq'})
            assigned = []
            for run in store.all('runs')[:50]:
                if node['id'] in run.get('participants', {}) and run.get('partition_hash'):
                    assigned.append({k: run[k] for k in ('id','name','spec','status','partition_hash','dataset_hash','capabilities','created_at')} |
                                    {'client_id': run['participants'][node['id']]['client_id'],
                                     'flower_address': f"{settings['advertise_host']}:{settings['flower_port']}"})
            return envelope({'runs': assigned, 'models': [clean_model(m) for m in store.all('models') if m['scope'] == 'global']})

    @app.get('/api/v1/agents/runs/{run_id}/partition')
    def agent_partition(run_id: str, node=Depends(agent)):
        run = store.get('runs', run_id)
        if not run or node['id'] not in run['participants'] or not run.get('partition_path'):
            raise HTTPException(404, 'Partition not available to this institution.')
        return FileResponse(run['partition_path'], media_type='application/json')

    @app.get('/api/v1/nodes', response_model=Envelope)
    def nodes(session=Depends(coordinator)):
        return envelope([node_view(n) for n in store.all('nodes')])

    @app.get('/api/v1/runs', response_model=Envelope)
    def runs(session=Depends(operator)):
        return envelope(store.all('runs') if service.mode == 'coordinator' else store.all('participations'))

    @app.post('/api/v1/runs', response_model=Envelope)
    def create_run(spec: RunSpec, session=Depends(coordinator)):
        return envelope(service.create_run(spec.model_dump()))

    @app.post('/api/v1/runs/{run_id}/start', response_model=Envelope)
    def start_run(run_id: str, session=Depends(coordinator)):
        return envelope(service.start_run(run_id))

    @app.post('/api/v1/runs/{run_id}/accept', response_model=Envelope)
    def accept(run_id: str, session=Depends(institution)):
        return envelope(clean_job(service.accept(run_id)))

    @app.post('/api/v1/runs/{run_id}/cancel', response_model=Envelope)
    def cancel(run_id: str, session=Depends(operator)):
        return envelope(service.cancel(run_id))

    @app.get('/api/v1/runs/{run_id}/events', response_model=Envelope)
    def events(run_id: str, session=Depends(operator)):
        kind = 'runs' if service.mode == 'coordinator' else 'participations'
        if store.get(kind, run_id) is None:
            raise HTTPException(404, 'Session not found')
        return envelope(store.events(run_id, limit=10000))

    @app.get('/api/v1/datasets', response_model=Envelope)
    def datasets(session=Depends(operator)):
        return envelope(store.all('datasets'))

    @app.post('/api/v1/datasets/prepare', response_model=Envelope)
    def prepare(body: DatasetRequest, session=Depends(operator)):
        if service.busy():
            raise Busy('Wait for this device to finish its current task.')
        payload = body.model_dump()
        if service.mode == 'institution':
            parts = [p for p in store.all('participations') if p['spec']['dataset'] == body.dataset and p.get('partition_path')]
            if parts:
                payload.update(partition_path=parts[0]['partition_path'], client_id=parts[0]['client_id'])
        return envelope(clean_job(service.task('download', payload)))

    @app.get('/api/v1/models', response_model=Envelope)
    def models(session=Depends(operator)):
        local = [clean_model(m) for m in store.all('models')]
        remote = store.get('settings', 'remote_models', {}).get('models', []) if service.mode == 'institution' else []
        institution_models = [m for n in store.all('nodes') for m in n.get('models', [])] if service.mode == 'coordinator' else []
        return envelope({'local': local, 'remote': remote, 'institution': institution_models})

    @app.post('/api/v1/models/import', response_model=Envelope)
    def import_model(body: ImportModel, session=Depends(operator)):
        if service.busy():
            raise Busy('Wait for the current task before validating another checkpoint.')
        path = contained_path(body.path, settings['artifact_roots'] + [str(service.state)])
        if not path.is_file() or path.suffix != '.pth':
            raise ValueError('Choose an existing project-generated .pth checkpoint.')
        if body.strategy in ('fedbn','fedbns'):
            raise ValueError('Import FedBN/FedBNS through a live institution session to verify local BatchNorm ownership.')
        return envelope(clean_job(service.task('model', body.model_dump() | {'path': str(path)})))

    @app.get('/api/v1/models/candidates', response_model=Envelope)
    def candidates(session=Depends(operator)):
        found = []
        for root in map(Path, settings['artifact_roots']):
            if not root.exists():
                continue
            for path in root.rglob('final_model.pth'):
                meta_path = path.parent/'fl_results.json'
                if not meta_path.exists():
                    continue
                try:
                    content = json.loads(meta_path.read_text(encoding='utf-8'))
                    meta = content['metadata']
                    dataset = next((d for d in ('bloodmnist','organamnist','pathmnist') if any(p == d or p.startswith(d+'_') for p in path.parts)), None)
                    strategy = meta.get('strategy', '').lower()
                    model = meta.get('model')
                    if not dataset or strategy not in ('fedavg','scaffold','scr') or model not in ('baseline_bn','baseline','cnn'):
                        continue
                    rounds = {int(r['round']) for r in content.get('rounds_history', []) if r.get('round')}
                    found.append({'path': str(path.resolve()), 'dataset': dataset, 'strategy': strategy, 'model': model,
                                  'source_run': path.parent.parent.parent.parent.name, 'source_round': max(rounds) if rounds else int(meta.get('num_rounds_requested') or 0),
                                  'selection_rule': 'final_round'})
                except (OSError, ValueError, KeyError, TypeError):
                    continue
        return envelope(sorted(found, key=lambda m: (m['dataset'],m['source_run']), reverse=True)[:100])

    @app.post('/api/v1/models/{model_id}/download', response_model=Envelope)
    def download_model(model_id: str, session=Depends(institution)):
        return envelope(clean_job(service.fetch_model(model_id)))

    @app.get('/api/v1/agents/models/{model_id}/file')
    def checkpoint_file(model_id: str, node=Depends(agent)):
        model = store.get('models', model_id)
        if not model or model['scope'] != 'global':
            raise HTTPException(404, 'Global model not found')
        return FileResponse(model['path'], filename=model_id+'.pth')

    @app.post('/api/v1/predict', response_model=Envelope)
    def predict(body: PredictRequest, session=Depends(institution)):
        if service.busy():
            raise Busy('This device is busy. Prediction is available after its current task finishes.')
        model = store.get('models', body.model_id)
        if not model or not model['verified']:
            raise ValueError('Select a verified model downloaded to this institution.')
        if model['scope'] == 'institution' and model['institution_id'] != settings['id']:
            raise HTTPException(403, 'This checkpoint belongs to another institution.')
        if (body.sample_index is None) == (body.image_base64 is None):
            raise ValueError('Choose either one benchmark sample or one uploaded image.')
        payload = body.model_dump() | {'manifest': model}
        parts = [p for p in store.all('participations') if p['spec']['dataset'] == model['dataset'] and p.get('partition_path')]
        if parts:
            partition = json.loads(Path(parts[0]['partition_path']).read_text())
            payload['allowed_indices'] = partition['clients'][str(parts[0]['client_id'])]['test_indices']
        return envelope(clean_job(service.task('predict', payload)))

    @app.get('/api/v1/experiments', response_model=Envelope)
    def experiments(session=Depends(coordinator)):
        return envelope(store.all('experiments'))

    @app.post('/api/v1/experiments/import', response_model=Envelope)
    def import_experiments(session=Depends(coordinator)):
        entries = scan(settings['artifact_roots'])
        for entry in entries:
            store.put('experiments', entry)
        return envelope({'count': len(entries)})

    frontend = ROOT / 'web' / 'dist'
    if (frontend / 'assets').is_dir():
        app.mount('/assets', StaticFiles(directory=frontend/'assets'), name='assets')

    @app.get('/{path:path}', include_in_schema=False)
    def spa(path: str):
        if path.startswith('api/'):
            raise HTTPException(404, 'API route not found')
        if path == 'mark.svg' and (frontend/'mark.svg').exists():
            return FileResponse(frontend/'mark.svg')
        if (frontend/'index.html').exists():
            return FileResponse(frontend/'index.html')
        return JSONResponse({'detail': 'Frontend has not been built. Run the web build from the setup guide.'}, status_code=503)

    return app
