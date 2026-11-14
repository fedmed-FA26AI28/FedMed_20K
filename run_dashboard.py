#!/usr/bin/env python3
"""FL Orchestrator Unified Dashboard Launcher.
Launches the modern CustomTkinter desktop dashboard for FL Server orchestration and FL Client execution.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Enforce UTF-8 standard output / error across platforms
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Windows High-DPI Scaling Initialization
if sys.platform.startswith("win"):
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def main():
    try:
        import customtkinter
    except ImportError:
        print("[!] Package 'customtkinter' is missing.")
        print("[*] Installing customtkinter...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "customtkinter"])

    from gui.app import main as launch_gui
    launch_gui()


if __name__ == "__main__":
    main()

