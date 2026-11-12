import os
import signal
import subprocess
import time
from pathlib import Path
import psutil
from portal.common import ROOT


def terminate_owned(pid, created):
    """Never terminate a recycled PID or an unrelated process tree."""
    try:
        parent = psutil.Process(pid)
        if abs(parent.create_time() - created) > 0.01:
            return
        targets = parent.children(recursive=True) + [parent]
        for child in reversed(targets):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(targets, timeout=3)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass


class Process:
    def __init__(self, argv, log_path, env):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self.log = open(log_path, 'a', encoding='utf-8')
        options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {'start_new_session': True}
        try:
            self.p = subprocess.Popen(argv, cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT,
                                      env=env, stdin=subprocess.DEVNULL, **options)
        except OSError:
            self.log.close()
            raise
        self.created = psutil.Process(self.p.pid).create_time()
        self.started = time.time()

    def poll(self):
        code = self.p.poll()
        if code is not None and not self.log.closed:
            self.log.close()
        return code

    def stop(self):
        terminate_owned(self.p.pid, self.created)
        self.p.wait(timeout=10)
        self.log.close()

    def identity(self):
        return {'pid': self.p.pid, 'process_created': self.created}
