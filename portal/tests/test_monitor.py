import copy
import csv
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from portal.app import create_app
from portal.monitor import classification, snapshot
from portal.tests.helpers import settings, PASSWORD


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        settings(self.root)
        self.app = create_app(self.root, start_service=False)
        self.store = self.app.state.service.store
        self.run = {'id':'run', 'status':'completed', 'spec':{'dataset':'bloodmnist'}, 'partition_hash':'partition'}
        self.store.put('runs', self.run)
        self.client = TestClient(self.app)
        self.client.headers['X-FedMed-Request'] = '1'

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def event(self, identity, phase, cid=0, rnd=1, **values):
        self.store.event(dict(id=identity, run_id='run', client_id=cid, round=rnd, phase=phase, time=1., **values))

    def test_deduplication_weighting_zero_accuracy_and_empty_test(self):
        for i in range(2):
            self.event('fit'+str(i), 'fit_completed', metrics={'num_samples':4,'train_loss':2.,'train_accuracy':.5,'upload_payload_kb':3.,'download_payload_kb':4.})
            self.event('eval'+str(i),'evaluation_completed',num_samples=4,accuracy=0.,loss=3.,confusion_matrix=[[0,2],[2,0]])
            self.event('epoch'+str(i),'epoch_completed',epoch=1,loss=2.)
            self.event('round'+str(i),'round_completed',cid=-1,metrics={'eval_accuracy':0.,'mcta':0.})
        self.event('empty','evaluation_skipped',cid=1,num_samples=0)
        report=snapshot(self.store,self.run,'coordinator')
        self.assertEqual(len(report['rounds']),1)
        self.assertEqual(len(report['epochs']),1)
        self.assertEqual(report['summary']['upload_payload_kb'],3.)
        self.assertEqual(report['summary']['bta'],0.)
        self.assertEqual(report['summary']['bmcta'],0.)
        self.assertEqual(report['evaluations'][0]['samples'],4)
        self.assertEqual(report['rounds'][0]['skipped_clients'],1)
        self.assertIsNone(report['client_rounds'][1]['eval_accuracy'])

    def test_classification_uses_counts_and_missing_class_semantics(self):
        result=classification([[3,1,0],[0,2,0],[0,0,0]],{'0':'A','1':'B','2':'C'})
        self.assertAlmostEqual(result['accuracy'],5/6)
        self.assertAlmostEqual(result['balanced_accuracy'],.875)
        self.assertAlmostEqual(result['macro_f1'],(6/7+4/5)/2)
        self.assertIsNone(result['classes'][2]['recall'])
        self.assertIsNone(result['classes'][2]['precision'])
        for bad in ([], [[0,0],[0,0]], [[-1]], [[float('nan')]], [[1,2]], 'wrong'):
            self.assertIsNone(classification(bad,{}))

    def test_log_relay_is_incremental_and_resources_use_worker_elapsed_time(self):
        import psutil
        folder=self.root/'job';folder.mkdir()
        log=folder/'process.log';log.write_text('first epoch\n')
        job={'id':'job','folder':str(folder),'context':{'run_id':'run'}}
        process=Mock()
        process.poll.return_value=None
        process.p.pid=os.getpid();process.created=psutil.Process().create_time();process.started=time.time()-3
        service=self.app.state.service
        service.capture_diagnostics(job,process)
        service.capture_diagnostics(job,process)
        with log.open('a') as stream:
            stream.write('Traceback: real worker error\n')
        process.poll.return_value=1
        service.capture_diagnostics(job,process)
        report=snapshot(self.store,self.run,'coordinator')
        self.assertEqual(len(report['logs']),2)
        self.assertIn('Traceback',report['logs'][-1]['text'])
        self.assertGreaterEqual(report['systems'][0]['worker_elapsed_seconds'],3)
        self.assertGreater(report['systems'][0]['process_tree_rss_mb'],0)

    def test_nonfinite_diagnostic_does_not_discard_the_epoch(self):
        from utils.portal_events import emit
        path=self.root/'nonfinite.jsonl'
        with patch.dict(os.environ,{'FEDMED_EVENT_LOG':str(path),'FEDMED_RUN_ID':'run','FEDMED_CLIENT_ID':'0'}):
            emit('epoch_completed',1,loss=0.5,last_batch_grad_l2=float('inf'))
        result=json.loads(path.read_text())
        self.assertEqual(result['loss'],0.5)
        self.assertIsNone(result['last_batch_grad_l2'])

    def test_restart_refuses_active_training_before_inspecting_processes(self):
        from portal.restart import stop_idle_portal
        self.store.update('runs','run',status='running')
        with patch('portal.restart.psutil.process_iter') as processes, patch('portal.restart.terminate_owned') as terminate:
            with self.assertRaisesRegex(ValueError,'busy'):
                stop_idle_portal(self.root)
            processes.assert_not_called();terminate.assert_not_called()
        self.store.update('runs','run',status='completed')
        self.store.put('jobs',{'id':'busy','status':'running'})
        with patch('portal.restart.terminate_owned') as terminate:
            with self.assertRaisesRegex(ValueError,'busy'):
                stop_idle_portal(self.root)
            terminate.assert_not_called()

    def test_restart_identifies_only_the_matching_portal_process(self):
        import socket
        import sys
        from portal.common import atomic_json
        from portal.processes import Process
        from portal.restart import stop_idle_portal
        from portal.storage import Store
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        state=self.root/'restart-instance'
        config=settings(state,port=port)
        config['trainer_python']=str(state/'missing-training-python')
        atomic_json(state/'settings.json',config)
        portal=Process([sys.executable,'-m','portal','serve','--state-dir',str(state),'--host','127.0.0.1'],self.root/'restart.log',os.environ.copy())
        unrelated=Process([sys.executable,'-c','import time; time.sleep(60)'],self.root/'unrelated.log',os.environ.copy())
        try:
            store=Store(state/'portal.sqlite3')
            deadline=time.time()+30
            while not any(j['status']=='failed' for j in store.all('jobs')):
                if portal.poll() is not None or time.time()>deadline:
                    self.fail('Isolated portal did not become idle: '+(self.root/'restart.log').read_text())
                time.sleep(.2)
            stop_idle_portal(state)
            self.assertIsNotNone(portal.poll())
            self.assertIsNone(unrelated.poll())
        finally:
            if portal.poll() is None:
                portal.stop()
            unrelated.stop()

    def test_authorization_filter_exports_and_more_than_10000_events(self):
        self.assertEqual(self.client.get('/api/v1/runs/run/monitor').status_code,401)
        self.client.post('/api/v1/session',json={'username':'operator','password':PASSWORD}).raise_for_status()
        with self.store.connection() as db:
            db.executemany('INSERT INTO events(id,run_id,data) VALUES(?,?,?)',[
                (f'log{i}','run',json.dumps(dict(id=f'log{i}',run_id='run',client_id=0,round=1,phase='process_log',time=i,text='=unsafe spreadsheet text')))
                for i in range(10005)])
        self.event('last','round_completed',cid=-1,metrics={'eval_accuracy':.75})
        self.event('other','epoch_completed',cid=1,epoch=1,loss=.5)
        report=self.client.get('/api/v1/runs/run/monitor?client_id=0').json()['data']
        self.assertEqual(report['summary']['current_accuracy'],.75)
        self.assertEqual(report['epochs'],[])
        self.assertEqual(len(report['logs']),250)
        self.assertEqual(report['limits']['logs_total'],10005)
        self.assertEqual(len(self.client.get('/api/v1/runs/run/events').json()['data']),2)
        self.assertEqual(len(self.client.get('/api/v1/runs/run/monitor/export').json()['logs']),10005)
        rows=list(csv.DictReader(io.StringIO(self.client.get('/api/v1/runs/run/monitor/export?format=csv').text)))
        self.assertEqual({r['record_type'] for r in rows},{'round','epoch'})
        settings(self.root/'institution','institution')
        with TestClient(create_app(self.root/'institution',start_service=False)) as other:
            other.headers['X-FedMed-Request']='1'
            other.post('/api/v1/session',json={'username':'operator','password':PASSWORD})
            self.assertEqual(other.get('/api/v1/runs/run/monitor').status_code,404)
            self.assertEqual(other.get('/api/v1/runs/run/monitor/export').status_code,404)

    def test_training_and_confusion_diagnostics_do_not_change_updates(self):
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        from client.client import _local_train, FedMedAIClient
        from models import get_parameters
        torch.set_num_threads(1)
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc=nn.Linear(3,2)
            def forward(self,x):
                return self.fc(x)
        torch.manual_seed(5)
        original=Tiny()
        images=torch.randn(12,3); labels=torch.arange(12)%2
        loader=DataLoader(TensorDataset(images,labels),batch_size=4,shuffle=False)
        for method in ('fedavg','scr'):
            models=[copy.deepcopy(original),copy.deepcopy(original)]
            logs=self.root/(method+'.jsonl')
            for model,enabled in zip(models,(False,True)):
                options={} if method=='fedavg' else dict(scr_sample_weights=torch.tensor([.5,1.5]),scr_anchor=torch.tensor([.8,.2]),scr_reference=(model.fc.weight.detach().clone(),model.fc.bias.detach().clone()),scr_head_lambda=.01)
                with patch.dict(os.environ,{'FEDMED_EVENT_LOG':str(logs) if enabled else '', 'FEDMED_RUN_ID':'run','FEDMED_CLIENT_ID':'0','FEDMED_ACTIVE_ROUND':'1'}):
                    _local_train(model,loader,torch.optim.SGD(model.parameters(),lr=.01),torch.device('cpu'),2,**options)
            for a,b in zip(models[0].parameters(),models[1].parameters()):
                self.assertTrue(torch.equal(a,b),f'{method}: telemetry changed model parameters')
            events=[json.loads(line) for line in logs.read_text().splitlines()]
            self.assertEqual(len(events),2)
            for e in events:
                self.assertAlmostEqual(e['loss'],e['data_objective_loss']+e['head_penalty_loss'],places=6)
                self.assertEqual(e['learning_rate'],.01)
                self.assertIsNone(e['gpu_memory_mb'])
            client=FedMedAIClient.__new__(FedMedAIClient)
            client.model=models[1];client.device=torch.device('cpu');client.test_loader=loader
            client.strategy=method;client.client_id=0;client.ping_logger=Mock()
            with patch.dict(os.environ,{'FEDMED_EVENT_LOG':str(logs),'FEDMED_RUN_ID':'run','FEDMED_CLIENT_ID':'0','FEDMED_LOCAL_CHECKPOINT':''}):
                loss,count,metrics=client.evaluate(get_parameters(client.model),{'server_round':1})
            e=json.loads(logs.read_text().splitlines()[-1])
            direct=torch.zeros(2,2,dtype=torch.int64)
            for true,pred in zip(labels,client.model(images).argmax(1)):
                direct[true,pred]+=1
            self.assertEqual(e['confusion_matrix'],direct.tolist())
            self.assertEqual(count,12)


if __name__=='__main__':
    unittest.main()
