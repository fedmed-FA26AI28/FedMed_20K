"""Validate saved research checkpoints through isolated coordinator/institution services.

Run only against the local smoke-test accounts. Result files are read without changes.
"""
import argparse
import base64
import io
import json
import time
from pathlib import Path
import httpx
from PIL import Image
from portal.tests.helpers import PASSWORD
from portal.tests.smoke import wait_until


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--coordinator',default='http://127.0.0.1:49687')
    parser.add_argument('--institution',default='http://127.0.0.1:49691')
    parser.add_argument('--output',default='.portal-qa/import-report.json')
    args=parser.parse_args()
    clients=[httpx.Client(base_url=url+'/api/v1',headers={'X-FedMed-Request':'1'},timeout=30,trust_env=False) for url in (args.coordinator,args.institution)]
    try:
        coordinator,institution=clients
        for client in clients:
            client.post('/session',json={'username':'operator','password':PASSWORD}).raise_for_status()
        candidates=coordinator.get('/models/candidates').json()['data']
        results=[]
        def task(client,response):
            response.raise_for_status();job=response.json()['data']
            job=wait_until(lambda:client.get('/jobs/'+job['id']).json()['data'],lambda j:j['status'] in ('completed','failed'),label='model worker')
            assert job['status']=='completed',job
            return job['result']
        for dataset in ('bloodmnist','organamnist','pathmnist'):
            candidate=next(c for c in candidates if c['dataset']==dataset and c['model']=='baseline_bn')
            model=task(coordinator,coordinator.post('/models/import',json=candidate))
            wait_until(lambda:institution.get('/models').json()['data']['remote'],lambda ms:any(m['id']==model['id'] for m in ms),label='model publication')
            local=task(institution,institution.post('/models/'+model['id']+'/download'))
            assert model['id']==local['id'] and model['checksum']==local['checksum']
            # A generated input checks the upload/inference path, not clinical accuracy.
            image=Image.new('RGB',(28,28),(90,140,190));stream=io.BytesIO();image.save(stream,format='PNG')
            result=task(institution,institution.post('/predict',json={'model_id':local['id'],'image_base64':base64.b64encode(stream.getvalue()).decode()}))
            assert len(result['top'])==3 and result['label'] in model['classes'].values()
            results.append({'dataset':dataset,'source_run':candidate['source_run'],'source_round':candidate['source_round'],'checkpoint':candidate['path'],
                            'status':'passed','checksum':model['checksum'],'model_id':model['id'],'classes':len(model['classes'])})
            print(dataset+': existing checkpoint imported, downloaded, verified, and used for local inference',flush=True)
        Path(args.output).write_text(json.dumps(results,indent=2),encoding='utf-8')
    finally:
        for client in clients: client.close()


if __name__=='__main__':
    main()
