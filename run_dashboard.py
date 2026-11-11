#!/usr/bin/env python3
"""FedMedAI Unified Dashboard Launcher.
Launches the modern CustomTkinter desktop dashboard for FL Server orchestration and FL Client execution.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
