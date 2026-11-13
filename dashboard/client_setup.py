"""Export a portable configuration for a CLI-only remote Flower client."""
import math
import re
import shlex

from dashboard.system import endpoint


def remote_client_setup(server_config, address, client_id):
    host, _ = endpoint(address)
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', host):
        raise ValueError('Use an IPv4 address or hostname for the server.')
    if host == '0.0.0.0':
        raise ValueError('Use the server LAN IP, not the 0.0.0.0 bind address.')
    value = float(client_id)
    if not math.isfinite(value) or int(value) != value or not 0 <= value < server_config['num_clients']:
        raise ValueError('Client ID must be an integer from 0 to total clients minus one.')
    cfg = dict(server_config)
    # Paths and device registration are local to each remote machine.
    for key in ('partition_path', 'output_dir', 'save_dir', 'active_clients', 'device_type'):
        cfg.pop(key, None)
    cfg.update(server_address=address.strip(), client_id=int(value), device='auto')
    command = shlex.join(['python', '-m', 'client.client', '--config', 'configs/client_dashboard.yaml',
                          '--client_id', str(int(value)), '--server_address', address.strip()])
    return cfg, command
