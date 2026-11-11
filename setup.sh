#!/usr/bin/env bash
set -e

# Change to script directory
cd "$(dirname "$0")"

echo "======================================================================"
echo "          FedMedAI Environment Setup Script (uv + PyTorch GPU)        "
echo "======================================================================"
echo ""

# 1. Locate or install uv
if command -v uv >/dev/null 2>&1; then
    UV_CMD="uv"
elif [ -f "$HOME/.local/bin/uv" ]; then
    UV_CMD="$HOME/.local/bin/uv"
elif [ -f "$HOME/.cargo/bin/uv" ]; then
    UV_CMD="$HOME/.cargo/bin/uv"
else
    echo "[*] uv not found. Installing uv via Astral installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    if [ -f "$HOME/.local/bin/uv" ]; then
        UV_CMD="$HOME/.local/bin/uv"
    elif [ -f "$HOME/.cargo/bin/uv" ]; then
        UV_CMD="$HOME/.cargo/bin/uv"
    else
        echo "[x] Could not install uv automatically. Please visit https://docs.astral.sh/uv/"
        exit 1
    fi
fi

echo "[+] Using uv: $UV_CMD"

# 2. Create .venv if not exists
if [ ! -f ".venv/bin/python" ]; then
    echo "[*] Creating virtual environment (.venv) using Python 3.10..."
    if ! "$UV_CMD" venv --python 3.10 .venv; then
        echo "[!] Python 3.10 not found directly, falling back to default python..."
        "$UV_CMD" venv .venv
    fi
    echo "[+] Virtual environment created."
else
    echo "[+] Existing .venv detected."
fi

# 3. Configure PyTorch backend
export UV_TORCH_BACKEND=auto
export UV_LINK_MODE=copy

# 4. Install requirements with GPU PyTorch auto-detection
echo "[*] Installing dependencies from requirements.txt..."
"$UV_CMD" pip install -r requirements.txt --python .venv/bin/python --torch-backend=auto --link-mode=copy

# 5. Guarantee PyTorch has CUDA wheels
echo "[*] Ensuring PyTorch with CUDA acceleration..."
"$UV_CMD" pip install torch torchvision torchaudio --python .venv/bin/python --torch-backend=auto --reinstall-package torch --reinstall-package torchvision --link-mode=copy

# 6. Verification
echo ""
echo "======================================================================"
echo "                     Environment Diagnostics                          "
echo "======================================================================"
.venv/bin/python -c "import torch; print('  [+] PyTorch Version : ' + str(torch.__version__)); print('  [+] CUDA Available  : ' + str(torch.cuda.is_available())); print('  [+] Active Device   : ' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (No CUDA device)'))"
.venv/bin/python -c "import flwr, customtkinter; print('  [+] Flower Version : ' + str(flwr.__version__)); print('  [+] CustomTkinter  : ' + str(customtkinter.__version__))"

echo ""
echo "======================================================================"
echo "[+] FedMedAI Environment Setup Complete!"
echo "[*] Activate your environment: source .venv/bin/activate"
echo "======================================================================"
