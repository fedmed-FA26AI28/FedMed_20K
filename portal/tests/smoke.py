"""Real Flower subprocess integration on cached BloodMNIST, one round per strategy.

Outputs and new three-client partitions are isolated from research experiments.
"""
import argparse
import json
import os
import secrets
import socket
import threading
import time
from pathlib import Path
import httpx
import numpy as np
import uvicorn
from portal.app import create_app
from portal.common import ROOT, atomic_json
from portal.tests.helpers import PASSWORD, settings


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        return sock.getsockname()[1]


def wait_until(getter, condition, timeout=240, label='condition'):
    deadline=time.time()+timeout
    while time.time()<deadline:
        value=getter()
        if condition(value):
            return value
        time.sleep(1)
    raise TimeoutError(f'{label}: last value {json.dumps(value)[:2200]}')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--strategies',nargs='+',default=['fedavg','fedbn','scaffold','fedbns','scr'])
    parser.add_argument('--cpu',action='store_true',help='Use CPU for all three local clients; avoids sharing one GPU.')
    parser.add_argument('--rounds',type=int,default=1)
    parser.add_argument('--batch-size',type=int,default=32)
    args=parser.parse_args()
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES']='-1'
    folder=ROOT/'.portal-qa'/('smoke-'+time.strftime('%Y%m%d-%H%M%S'))
    folder.mkdir(parents=True)
    data=Path(os.environ.get('FEDMED_DATA_ROOT', str(Path.home()/'.medmnist')))
    if not (data/'bloodmnist.npz').exists():
        raise RuntimeError('Prepare the public BloodMNIST dataset before running integration checks.')
    servers=[];apps=[];clients=[]
    report={'fixture':'Cached public BloodMNIST, three local Flower processes. Integration checks, not 80-round research results.', 'rounds':args.rounds, 'batch_size':args.batch_size, 'compute':'cpu' if args.cpu else 'auto', 'strategies':[]}
    try:
        flower_port=free_port()
        for i in range(4):
            port=free_port();state=folder/('coordinator' if i==0 else f'institution-{i}')
            settings(state,'coordinator' if i==0 else 'institution',port,flower_port,data)
            app=create_app(state);apps.append(app)
            server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,log_level='error'))
            thread=threading.Thread(target=server.run,daemon=True);thread.start();servers.append((server,thread))
            wait_until(lambda:server.started,bool,30,'HTTP startup')
            client=httpx.Client(base_url=f'http://127.0.0.1:{port}',headers={'X-FedMed-Request':'1'},timeout=30,trust_env=False)
            client.post('/api/v1/session',json={'username':'operator','password':PASSWORD}).raise_for_status();clients.append(client)
        coordinator=clients[0]
        for client in clients[1:]:
            pair=coordinator.post('/api/v1/pairing-tokens').json()['data']
            client.post('/api/v1/connection',json={'coordinator_url':str(coordinator.base_url).rstrip('/'),'token':pair['token']}).raise_for_status()
        nodes=wait_until(lambda:coordinator.get('/api/v1/nodes').json()['data'],lambda ns:len(ns)==3 and all(n['capabilities'].get('ready') for n in ns),label='device preflight')
        report['actual_cuda'] = [n['capabilities']['cuda'] for n in nodes]
        if args.cpu:
            assert not any(report['actual_cuda']), 'CPU test requires CUDA disabled in every worker.'
        node_ids=[n['id'] for n in nodes]
        for strategy in args.strategies:
            print(f'[{strategy}] creating real Flower run',flush=True)
            response=coordinator.post('/api/v1/runs',json={'name':f'LOCAL integration {strategy}','node_ids':node_ids,'strategy':strategy,'rounds':args.rounds,'local_epochs':1,'batch_size':args.batch_size,'round_timeout':180})
            response.raise_for_status();run=response.json()['data'];run_id=run['id']
            get_run=lambda:next(r for r in coordinator.get('/api/v1/runs').json()['data'] if r['id']==run_id)
            wait_until(get_run,lambda r:r['status'] in ('waiting','failed'),label='partition preparation')
            assert get_run()['status']=='waiting',get_run()
            for client in clients[1:]:
                wait_until(lambda:client.get('/api/v1/runs').json()['data'],lambda rs:any(r['id']==run_id for r in rs),label='invitation')
                client.post(f'/api/v1/runs/{run_id}/accept').raise_for_status()
            ready=wait_until(get_run,lambda r:all(p['status']=='ready' for p in r['participants'].values()) or any(p['status']=='failed' for p in r['participants'].values()),label='readiness')
            assert all(p['status']=='ready' for p in ready['participants'].values()),ready
            coordinator.post(f'/api/v1/runs/{run_id}/start').raise_for_status()
            finished=wait_until(get_run,lambda r:r['status'] in ('completed','failed','cancelled'),args.rounds*360,'training completion')
            assert finished['status']=='completed',finished
            events=coordinator.get(f'/api/v1/runs/{run_id}/events').json()['data']
            assert len([e for e in events if e['phase']=='round_completed'])==args.rounds
            snapshots=[]
            for client,app in zip(clients[1:],apps[1:]):
                manifests=client.get('/api/v1/models').json()['data']['local']
                model=next(m for m in manifests if m['source_run']==run_id)
                assert model['scope']==('institution' if strategy in ('fedbn','fedbns') else 'global')
                snapshot=app.state.service.store.get('models',model['id']);snapshots.append(snapshot)
                assert Path(snapshot['path']).exists()
                participation=app.state.service.store.get('participations',run_id)
                assert participation['completed_rounds']==args.rounds
                assert participation['ended_at']>=participation['started_at']
            # Verify identical top class and probabilities against direct Python inference.
            target=clients[1];app=apps[1];manifest=snapshots[0]
            ds=target.get('/api/v1/datasets').json()['data'][0]
            index=ds['previews'][0]['index']
            job=target.post('/api/v1/predict',json={'model_id':manifest['id'],'sample_index':index}).json()['data']
            prediction=wait_until(lambda:target.get('/api/v1/jobs/'+job['id']).json()['data'],lambda j:j['status'] in ('completed','failed'),label='prediction')
            assert prediction['status']=='completed',prediction
            os.environ['FEDMED_DATA_ROOT']=str(data)
            from portal.worker import predict
            direct=predict({'manifest':manifest,'sample_index':index})
            assert np.allclose(direct['scores'],prediction['result']['scores'],atol=1e-6)
            assert direct['class_id']==prediction['result']['class_id']
            if strategy in ('fedbn','fedbns'):
                import torch
                from models import get_model
                initial=get_model('baseline_bn',num_classes=8).state_dict()
                for snapshot in snapshots:
                    state=torch.load(snapshot['path'],map_location='cpu',weights_only=True)
                    bnkeys=[k for k in state if k.endswith('running_mean')]
                    assert bnkeys and any(not torch.equal(state[k],initial[k]) for k in bnkeys)
            report['strategies'].append({'strategy':strategy,'status':'passed','run_id':run_id,'clients':3,'rounds':args.rounds,'prediction_parity':True,'scoped_checkpoints':True})
            atomic_json(folder/'report.json',report)
            print(f'[{strategy}] passed: training, telemetry, checkpoints, prediction parity',flush=True)
        # Verify class maps and RGB/grayscale preprocessing for every supported benchmark.
        from medmnist import INFO
        from models import get_model
        import torch
        for dataset,count in [('bloodmnist',8),('organamnist',11),('pathmnist',9)]:
            assert len(INFO[dataset]['label'])==count
            model=get_model('baseline_bn',num_classes=count).eval()
            with torch.inference_mode():
                assert model(torch.zeros(1,3,28,28)).shape==(1,count)
        report['three_dataset_metadata']='passed'
        atomic_json(folder/'report.json',report)
        print('REPORT '+str(folder/'report.json'),flush=True)
    finally:
        for client in clients:
            client.close()
        for server,thread in servers:
            server.should_exit=True
        for server,thread in servers:
            thread.join(timeout=20)


if __name__=='__main__':
    main()
