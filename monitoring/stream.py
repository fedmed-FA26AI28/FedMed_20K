"""Small, dependency-free events consumed by the NiceGUI dashboard."""
import json
import math
import os
import signal

NODE_FIELDS = ('client_id', 'device_ipv4', 'device_type', 'ping_ms', 'cpu_percent',
               'ram_percent', 'ram_used_mb', 'gpu_memory_mb', 'train_loss', 'train_accuracy',
               'num_samples', 'local_epochs', 'epochs_run', 'learning_rate', 'training_time',
               'epoch_time_avg', 'weight_size_kb')


def install_interrupt_handler():
    """Map Windows process-group CTRL_BREAK to the existing Ctrl+C cleanup paths."""
    if os.name == 'nt' and os.environ.get('FEDMED_DASHBOARD_STREAM') == '1':
        signal.signal(signal.SIGBREAK, signal.default_int_handler)


def emit_event(kind: str, **payload) -> None:
    # Opt-in for subprocesses started by the dashboard; CLI logs stay unchanged.
    if os.environ.get('FEDMED_DASHBOARD_STREAM') != '1':
        return

    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    print('[METRIC] ' + json.dumps(clean({'kind': kind, **payload}), ensure_ascii=True), flush=True)
