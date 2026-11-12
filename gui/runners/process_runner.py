import os
import sys
import subprocess
import threading
from typing import List, Callable, Optional
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class SubprocessRunner:
    """Manages asynchronous subprocess execution, background stream reading, and clean termination."""

    def __init__(
        self,
        on_log: Optional[Callable[[str], None]] = None,
        on_exit: Optional[Callable[[int], None]] = None,
    ):
        self.on_log = on_log
        self.on_exit = on_exit
        self.process: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._is_running = False

    def is_running(self) -> bool:
        if self.process is None:
            return False
        return self.process.poll() is None

    def start(self, cmd: List[str], env_vars: Optional[dict] = None) -> bool:
        """Start subprocess with non-blocking line-by-line stdout capture."""
        if self.is_running():
            return False

        full_env = os.environ.copy()
        if env_vars:
            full_env.update(env_vars)
        # Ensure PYTHONPATH includes project root
        full_env["PYTHONPATH"] = str(PROJECT_ROOT)
        full_env["PYTHONUNBUFFERED"] = "1"

        try:
            # On Windows, use CREATE_NEW_PROCESS_GROUP to allow graceful signal handling if needed
            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

            self.process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=full_env,
                text=True,
                bufsize=1,
                universal_newlines=True,
                creationflags=creationflags,
            )
            self._is_running = True
            self._reader_thread = threading.Thread(target=self._read_stream, daemon=True)
            self._reader_thread.start()
            return True
        except Exception as e:
            if self.on_log:
                self.on_log(f"[ERROR] Failed to start process: {e}\n")
            return False

    def _read_stream(self):
        """Continuously drain process stdout until termination."""
        if not self.process or not self.process.stdout:
            return

        try:
            for line in iter(self.process.stdout.readline, ""):
                if not line and self.process.poll() is not None:
                    break
                if self.on_log:
                    self.on_log(line)
        except Exception as e:
            if self.on_log:
                self.on_log(f"[ERROR] Stream read exception: {e}\n")
        finally:
            self._is_running = False
            return_code = self.process.poll() if self.process else -1
            if self.on_exit:
                try:
                    self.on_exit(return_code or 0)
                except Exception:
                    pass

    def stop(self) -> bool:
        """Cleanly terminate or kill the running subprocess tree."""
        if not self.is_running() or not self.process:
            return True

        pid = self.process.pid
        try:
            if sys.platform == "win32":
                # Taskkill with tree kill flag /T
                subprocess.run(
                    f"taskkill /F /T /PID {pid}",
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
        except Exception as e:
            if self.on_log:
                self.on_log(f"[ERROR] Failed killing process PID {pid}: {e}\n")

        self._is_running = False
        return True
