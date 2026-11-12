"""Optional JSONL telemetry; no web dependencies and no effect on training math."""
import json
import math
import os
import time
import uuid
from pathlib import Path


def _finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite_json(item) for item in value]
    return value


def emit(phase, round=0, **values):
    target = os.environ.get('FEDMED_EVENT_LOG')
    if not target:
        return
    event = dict(id=uuid.uuid4().hex, run_id=os.environ.get('FEDMED_RUN_ID', ''),
                 client_id=int(os.environ.get('FEDMED_CLIENT_ID', '-1')),
                 time=time.time(), round=int(round or 0), phase=phase, **values)
    try:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(_finite_json(event), allow_nan=False) + '\n')
    except (OSError, ValueError) as exc:
        print(f'[Portal] Could not record telemetry: {exc}', flush=True)
