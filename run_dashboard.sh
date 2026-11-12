#!/usr/bin/env bash
# FL Orchestrator & Live Monitor Launcher

set -e

# Navigate to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "          FL Orchestrator & Live Monitor"
echo "======================================================================"
echo ""

# 1. Check local virtual environment inside project root
if [ -f ".venv/bin/python" ]; then
    PYTHON_EXE=".venv/bin/python"
elif [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON_EXE=".venv/Scripts/python.exe"
elif [ -f "../.venv/bin/python" ]; then
    PYTHON_EXE="../.venv/bin/python"
elif [ -f "../.venv/Scripts/python.exe" ]; then
    PYTHON_EXE="../.venv/Scripts/python.exe"
elif command -v python3 &> /dev/null; then
    PYTHON_EXE="python3"
elif command -v python &> /dev/null; then
    PYTHON_EXE="python"
else
    echo "[!] No Python interpreter found on PATH or in .venv."
    exit 1
fi

echo "[*] Launching dashboard with: $PYTHON_EXE"
echo ""

exec "$PYTHON_EXE" run_dashboard.py "$@"
