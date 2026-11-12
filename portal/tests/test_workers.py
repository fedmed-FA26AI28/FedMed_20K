"""Training-environment correctness checks; generated weights are fixtures, not research models."""
import base64
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
import numpy as np
import torch
from PIL import Image
from medmnist import INFO
from models import get_model
from portal.worker import model_manifest, predict
from datasets.medmnist_code import get_evaluation_transform


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def checkpoint(self, dataset='bloodmnist', strategy='scr'):
        model=get_model('baseline_bn',num_classes=len(INFO[dataset]['label'])).eval()
        path=self.root/(dataset+'.pth')
        torch.save(model.state_dict(),path)
        request={'path':str(path),'dataset':dataset,'model':'baseline_bn','strategy':strategy,
                 'source_run':'test-fixture','source_round':0,'selection_rule':'final_round'}
        return model,path,request

    def test_prediction_matches_direct_python_for_all_dataset_transforms(self):
        for dataset,classes in [('bloodmnist',8),('organamnist',11),('pathmnist',9)]:
            with self.subTest(dataset=dataset):
                model,path,request=self.checkpoint(dataset)
                manifest=model_manifest(request)
                self.assertEqual(len(manifest['classes']),classes)
                self.assertEqual(manifest['classes'],INFO[dataset]['label'])
                array=np.arange(28*28*3,dtype=np.uint8).reshape(28,28,3)
                source=Image.fromarray(array)
                stream=io.BytesIO();source.save(stream,format='PNG')
                encoded=base64.b64encode(stream.getvalue()).decode()
                # Explicit reference preprocessing, independent of worker helper.
                pixels=np.asarray(source.convert('L' if dataset=='organamnist' else 'RGB'),dtype=np.float32)/255
                if pixels.ndim==2:
                    pixels=np.repeat(pixels[:,:,None],3,axis=2)
                tensor=torch.from_numpy(pixels.transpose(2,0,1)).sub(.5).div(.5)
                with torch.inference_mode():
                    logits=model(tensor.unsqueeze(0));scores=logits.softmax(1)[0]
                result=predict({'manifest':manifest,'image_base64':encoded})
                self.assertEqual(result['class_id'],int(logits.argmax(1)))
                np.testing.assert_allclose(result['scores'],scores.numpy(),atol=1e-6)
                transformed=get_evaluation_transform(dataset)(source.convert('L' if dataset=='organamnist' else 'RGB'))
                torch.testing.assert_close(transformed,tensor)

    def test_incompatible_or_changed_checkpoints_rejected(self):
        _,path,request=self.checkpoint()
        manifest=model_manifest(request)
        with self.assertRaises(RuntimeError):
            model_manifest(dict(request,dataset='pathmnist'))
        with self.assertRaisesRegex(ValueError,'institution checkpoint'):
            model_manifest(dict(request,strategy='fedbn'))
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'checksum'):
            predict({'manifest':manifest,'image_base64':''})

    def test_empty_evaluation_saves_complete_local_bn_and_skips_metrics(self):
        from client.client import FedMedAIClient
        from models import get_parameters
        for strategy in ('fedbn','fedbns'):
            with self.subTest(strategy=strategy):
                client=FedMedAIClient.__new__(FedMedAIClient)
                client.model=get_model('baseline_bn',num_classes=8)
                client.test_loader=[];client.device=torch.device('cpu')
                client.client_id=9;client.strategy=strategy;client.ping_logger=Mock()
                for key,value in client.model.state_dict().items():
                    if key.endswith('running_mean'):
                        value.fill_(.37)
                global_model=get_model('baseline_bn',num_classes=8)
                checkpoint=self.root/(strategy+'.pth');events=self.root/(strategy+'.jsonl')
                with patch.dict(os.environ,{'FEDMED_LOCAL_CHECKPOINT':str(checkpoint),'FEDMED_EVENT_LOG':str(events),'FEDMED_RUN_ID':'empty','FEDMED_CLIENT_ID':'9'}):
                    loss,count,metrics=client.evaluate(get_parameters(global_model),{'server_round':1,'strategy':strategy})
                self.assertEqual(count,0);self.assertEqual(loss,0)
                self.assertEqual(metrics['client_id'],9)
                saved=torch.load(checkpoint,weights_only=True)
                for key,value in client.model.state_dict().items():
                    torch.testing.assert_close(saved[key],value)
                    if key.endswith('running_mean'):
                        torch.testing.assert_close(saved[key],torch.full_like(value,.37))
                self.assertIn('evaluation_skipped',[json.loads(line)['phase'] for line in events.read_text().splitlines()])

    def test_strict_roster_rejects_missing_client_results(self):
        from server.server import FedMedAIServer, InterruptibleClientManager
        from flwr.server.strategy import FedAvg
        from flwr.common import ndarrays_to_parameters
        parameters=ndarrays_to_parameters([np.zeros(1,dtype=np.float32)])
        strategy=FedAvg(min_fit_clients=3,min_available_clients=3)
        server=FedMedAIServer(client_manager=InterruptibleClientManager(),strategy=strategy)
        # Flower returned two successes and one failed/disconnected client.
        with patch.dict(os.environ,{'FEDMED_STRICT_ROSTER':'1','FEDMED_ROSTER_TIMEOUT':'0'}):
            self.assertFalse(server.client_manager().wait_for(3))
            with patch.object(server,'_get_initial_parameters',return_value=parameters), patch.object(server,'fit_round',return_value=(parameters,{},([object(),object()],[TimeoutError('client disconnected')]))):
                with self.assertRaisesRegex(RuntimeError,'roster'):
                    server.fit(num_rounds=1,timeout=1)

    def test_round_callback_does_not_double_communication(self):
        from types import SimpleNamespace
        from monitoring.metrics import FLMetricsRecorder
        recorder=FLMetricsRecorder()
        result=SimpleNamespace(num_examples=10,metrics={'client_id':0,'weight_size_kb':12,'train_loss':.4,'train_accuracy':.8})
        recorder.record_fit_results(1,2,[(SimpleNamespace(cid='0'),result)],{})
        recorder.record_fit_results(1,2,[(SimpleNamespace(cid='0'),result)],{})
        self.assertEqual(len(recorder.round_records),1)
        self.assertEqual(len(recorder.client_records),1)
        self.assertEqual(recorder.round_records[0]['total_weight_size_kb'],12)


if __name__=='__main__':
    unittest.main()
