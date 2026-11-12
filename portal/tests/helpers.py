import sys
import uuid
from pathlib import Path
from portal.common import ROOT, atomic_json, password_hash

PASSWORD = 'test-only-password'


def settings(state, mode='coordinator', port=8000, flower_port=8080, data_root=None):
    value = dict(id=uuid.uuid4().hex, mode=mode, name='Local verification '+state.name,
                 username='operator', password_hash=password_hash(PASSWORD), trainer_python=sys.executable,
                 data_root=str(data_root or state/'data'), advertise_host='127.0.0.1', port=port,
                 flower_port=flower_port, device_type='pc', artifact_roots=[str(ROOT/'results'), str(ROOT.parent/'results')])
    atomic_json(state/'settings.json', value)
    return value
