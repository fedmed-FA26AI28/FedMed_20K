"""Operator-triggered restart of an idle local portal; never cancels training."""
import os
from pathlib import Path
import psutil
from portal.common import ROOT
from portal.processes import terminate_owned
from portal.schemas import ACTIVE
from portal.storage import Store


def stop_idle_portal(state):
    state = Path(state).resolve()
    if not (state / 'settings.json').exists():
        raise ValueError('Portal instance does not exist.')
    store = Store(state / 'portal.sqlite3')

    def require_idle():
        running = [r for kind in ('runs', 'participations') for r in store.all(kind)
                   if r.get('status') in ACTIVE | {'training', 'checking', 'ready'}]
        jobs = [j for j in store.all('jobs') if j.get('status') in ('queued', 'running')]
        if running or jobs:
            raise ValueError('Portal is busy. Wait for the session and compute tasks to finish, or cancel through the website, then retry. No processes were stopped.')

    require_idle()
    current = psutil.Process(os.getpid())
    excluded = {current.pid, *(p.pid for p in current.parents())}
    owned = {}
    for process in psutil.process_iter(['pid', 'name']):
        if process.pid in excluded or not (process.info['name'] or '').lower().startswith('python'):
            continue
        try:
            args = process.cmdline()
            if len(args) < 4 or args[1:3] != ['-m', 'portal'] or args[3] not in ('serve', 'restart'):
                continue
            cwd = Path(process.cwd()).resolve()
            folder = args[args.index('--state-dir') + 1] if '--state-dir' in args else '.portal'
            if cwd == ROOT and (cwd / folder).resolve() == state:
                owned[process.pid] = (process, process.create_time())
        except (psutil.NoSuchProcess, psutil.AccessDenied, IndexError):
            continue
    roots = [(p, created) for p, created in owned.values() if p.ppid() not in owned]
    def console_host(process):
        # CREATE_NO_WINDOW may attach an owned Windows console-host helper.
        try:
            return os.name == 'nt' and Path(process.exe()).resolve() == (Path(os.environ['SystemRoot'])/'System32'/'conhost.exe').resolve()
        except (psutil.AccessDenied, psutil.NoSuchProcess, KeyError):
            return False
    for parent, _ in roots:
        other = [child for child in parent.children(recursive=True) if child.pid not in owned and not console_host(child)]
        if other:
            raise ValueError('The portal still owns a worker process (' + ', '.join(p.name() for p in other) + '). Wait for it to finish before restarting.')
    require_idle()
    for parent, created in roots:
        terminate_owned(parent.pid, created)
    for pid, (_, created) in owned.items():
        try:
            if abs(psutil.Process(pid).create_time()-created) < .01:
                raise ValueError('Could not stop the existing portal. Close its terminal and retry.')
        except psutil.NoSuchProcess:
            pass
