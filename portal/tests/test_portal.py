import json
import secrets
import tempfile
import time
import unittest
import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient
from portal.app import create_app
from portal.common import token_hash, contained_path
from portal.history import unique_rounds
from portal.schemas import RunSpec
from portal.storage import Store
from portal.tests.helpers import PASSWORD, settings
from portal.processes import Process, terminate_owned
from portal.common import connectivity


class PortalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        settings(self.path)
        self.app = create_app(self.path, start_service=False)
        self.client = TestClient(self.app)
        self.client.headers['X-FedMed-Request'] = '1'

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def login(self):
        r = self.client.post('/api/v1/session', json={'username':'operator','password':PASSWORD})
        self.assertEqual(r.status_code, 200)

    def test_authentication_csrf_and_workspace_permissions(self):
        self.assertEqual(self.client.get('/api/v1/overview').status_code, 401)
        self.assertEqual(self.client.post('/api/v1/session', headers={'Origin':'http://attacker.example'}, json={'username':'operator','password':PASSWORD}).status_code, 403)
        self.login()
        self.assertEqual(self.client.get('/api/v1/overview').status_code, 200)
        self.assertEqual(self.client.post('/api/v1/predict', json={'model_id':'missing','sample_index':0}).status_code, 403)
        self.assertEqual(self.client.delete('/api/v1/session').status_code, 200)
        self.assertEqual(self.client.get('/api/v1/overview').status_code, 401)

    def test_pairing_token_single_use_and_event_scope(self):
        self.login()
        token = self.client.post('/api/v1/pairing-tokens').json()['data']['token']
        body = {'token':token,'name':'Institution A','instance_id':'node-a','device_type':'pc'}
        paired = self.client.post('/api/v1/agents/pair', json=body)
        self.assertEqual(paired.status_code, 200)
        self.assertEqual(self.client.post('/api/v1/agents/pair', json=body).status_code, 403)
        bearer = {'Authorization':'Bearer '+paired.json()['data']['token']}
        store = self.app.state.service.store
        store.put('runs', {'id':'run-a','status':'waiting','participants':{'node-a':{'client_id':0,'status':'invited'}}})
        event = {'id':'event-a','run_id':'run-a','client_id':0,'round':1,'time':time.time(),'phase':'training'}
        for _ in range(2):
            self.assertEqual(self.client.post('/api/v1/agents/heartbeat', headers=bearer, json={'events':[event]}).status_code, 200)
        self.assertEqual(len(store.events('run-a')),1)
        other = dict(event,id='event-b',client_id=1)
        self.client.post('/api/v1/agents/heartbeat', headers=bearer, json={'events':[other]})
        self.assertEqual(len(store.events('run-a')),1)
        self.assertEqual(self.client.get('/api/v1/agents/runs/not-owned/partition',headers=bearer).status_code,404)

    def test_validation_and_no_arbitrary_paths(self):
        self.login()
        response = self.client.post('/api/v1/runs', json={'node_ids':['node-a','node-a']})
        self.assertEqual(response.status_code,422)
        self.assertEqual(self.client.post('/api/v1/runs', json={'node_ids':['missing']}).status_code,400)
        with self.assertRaises(ValueError):
            contained_path(self.path.parent/'secret', [self.path])
        with self.assertRaises(ValueError):
            RunSpec(node_ids=['x'], learning_rate=float('nan'))

    def test_import_preserves_evaluation_and_deduplicates_payload(self):
        rows = unique_rounds([{'round':'1','eval_accuracy':'0.8','total_weight_size_kb':'1024'},
                              {'round':'1','eval_accuracy':'','total_weight_size_kb':'1024'},
                              {'round':'2','eval_accuracy':'0.9','total_weight_size_kb':'1024'}])
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['eval_accuracy'],'0.8')
        self.assertEqual(sum(float(r['total_weight_size_kb']) for r in rows),2048)

    def test_restart_interrupts_stale_run(self):
        store=self.app.state.service.store
        store.put('runs', {'id':'stale', 'status':'finalizing'})
        service=self.app.state.service
        service.start()
        service.close()
        self.assertEqual(store.get('runs','stale')['status'],'interrupted')

    def test_institution_cannot_read_coordinator_operations(self):
        state=self.path/'institution'
        settings(state,'institution')
        with TestClient(create_app(state,start_service=False)) as client:
            client.headers['X-FedMed-Request']='1'
            client.post('/api/v1/session',json={'username':'operator','password':PASSWORD})
            self.assertEqual(client.get('/api/v1/nodes').status_code,403)
            self.assertEqual(client.post('/api/v1/runs',json={'node_ids':['x']}).status_code,403)
            self.assertEqual(client.get('/api/v1/runs/private/events').status_code,404)

    def test_cancel_stops_only_owned_process_and_preserves_history(self):
        service = self.app.state.service
        spec = RunSpec(node_ids=['one']).model_dump()
        service.store.put('runs', {'id':'cancel-test','status':'running','spec':spec})
        job = service.training_job('cancel-test', spec)
        unrelated = Process([sys.executable,'-c','import time; time.sleep(60)'], self.path/'unrelated.log', os.environ.copy())
        try:
            service.launch(job,[sys.executable,'-c','import time; time.sleep(60)'])
            owned = service.processes[job['id']]
            terminate_owned(owned.p.pid, owned.created+10)  # A recycled PID must be ignored.
            self.assertIsNone(owned.poll())
            service.cancel('cancel-test')
            self.assertIsNotNone(owned.poll())
            self.assertIsNone(unrelated.poll())
            self.assertEqual(service.store.get('runs','cancel-test')['status'],'cancelled')
            self.assertEqual(service.store.get('jobs',job['id'])['status'],'cancelled')
        finally:
            unrelated.stop()

    def test_failed_client_fails_fixed_roster_and_connection_thresholds(self):
        service = self.app.state.service
        service.store.put('runs',{'id':'disconnect','status':'running','participants':{'one':{'status':'failed'},'two':{'status':'training'}}})
        service.coordinator_tick()
        self.assertEqual(service.store.get('runs','disconnect')['status'],'failed')
        self.assertEqual(connectivity(time.time()-10),'online')
        self.assertEqual(connectivity(time.time()-20),'stale')
        self.assertEqual(connectivity(time.time()-31),'offline')

    def test_invalid_python_fails_job_instead_of_retry_loop(self):
        service = self.app.state.service
        service.settings['trainer_python'] = str(self.path/'missing-python.exe')
        job=service.task('preflight',{})
        service.poll_jobs()
        self.assertEqual(service.store.get('jobs',job['id'])['status'],'failed')
        self.assertFalse(service.store.get('settings','capabilities')['ready'])

    def test_missing_and_wrong_institution_model_rejected(self):
        state=self.path/'institution'
        settings(state,'institution')
        app=create_app(state,start_service=False)
        with TestClient(app) as client:
            client.headers['X-FedMed-Request']='1'
            client.post('/api/v1/session',json={'username':'operator','password':PASSWORD})
            self.assertEqual(client.post('/api/v1/predict',json={'model_id':'missing','sample_index':0}).status_code,400)
            app.state.service.store.put('models', {'id':'foreign','verified':True,'scope':'institution','institution_id':'another-device'})
            self.assertEqual(client.post('/api/v1/predict',json={'model_id':'foreign','sample_index':0}).status_code,403)


if __name__ == '__main__':
    unittest.main()
