@echo off
setlocal EnableDelayedExpansion
title FedMedAI Environment Setup
cd /d "%~dp0"

echo ======================================================================
echo           FedMedAI Environment Setup (uv + PyTorch GPU)
echo ======================================================================
echo.

:: 1. Locate or install uv
echo [*] Checking for uv package manager...
set "UV_CMD="
where uv >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set "UV_CMD=uv"
) else if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "UV_CMD=%USERPROFILE%\.local\bin\uv.exe"
) else if exist "%USERPROFILE%\.cargo\bin\uv.exe" (
    set "UV_CMD=%USERPROFILE%\.cargo\bin\uv.exe"
)

if "%UV_CMD%"=="" (
    echo [!] uv was not found on your system.
    echo [*] Downloading and installing uv via Astral standalone installer...
    powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if exist "%USERPROFILE%\.local\bin\uv.exe" (
        set "UV_CMD=%USERPROFILE%\.local\bin\uv.exe"
    ) else if exist "%USERPROFILE%\.cargo\bin\uv.exe" (
        set "UV_CMD=%USERPROFILE%\.cargo\bin\uv.exe"
    ) else (
        echo [x] Failed to install uv automatically. Please install uv manually: https://docs.astral.sh/uv/
        pause
        exit /b 1
    )
)

echo [+] Using uv: %UV_CMD%
echo.

:: 2. Create or verify virtual environment in .venv
if not exist ".venv\Scripts\python.exe" (
    echo [*] Creating Python 3.10 virtual environment in .venv...
    "%UV_CMD%" venv --python 3.10 .venv
    if %ERRORLEVEL% neq 0 (
        echo [!] Python 3.10 not found directly, falling back to system default python...
        "%UV_CMD%" venv .venv
    )
    if not exist ".venv\Scripts\python.exe" (
        echo [x] Failed to create .venv virtual environment.
        pause
        exit /b 1
    )
    echo [+] Created .venv successfully.
) else (
    echo [+] Existing .venv detected.
)
echo.

:: 3. Set environment variables to enforce CUDA detection
set "UV_TORCH_BACKEND=auto"
set "UV_LINK_MODE=copy"

:: 4. Install all dependencies from requirements.txt with auto GPU backend
echo [*] Installing dependencies from requirements.txt with CUDA auto-backend...
"%UV_CMD%" pip install -r requirements.txt --python .venv\Scripts\python.exe --torch-backend=auto --link-mode=copy
if %ERRORLEVEL% neq 0 (
    echo [!] Warning: Some dependencies had warnings, proceeding to verify PyTorch GPU...
)

:: 5. Guarantee PyTorch GPU / CUDA wheels are installed (force reinstall package if CPU was present)
echo.
echo [*] Ensuring PyTorch has CUDA acceleration enabled...
"%UV_CMD%" pip install torch torchvision torchaudio --python .venv\Scripts\python.exe --torch-backend=auto --reinstall-package torch --reinstall-package torchvision --link-mode=copy
if %ERRORLEVEL% neq 0 (
    echo [x] Failed to install PyTorch with CUDA backend.
    pause
    exit /b 1
)

:: 6. Verification and Diagnostics
echo.
echo ======================================================================
echo                     Environment Diagnostics
echo ======================================================================
".venv\Scripts\python.exe" -c "import torch; print('  [+] PyTorch Version : ' + str(torch.__version__)); print('  [+] CUDA Available  : ' + str(torch.cuda.is_available())); print('  [+] Active Device   : ' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (No CUDA device)'))"
".venv\Scripts\python.exe" -c "import flwr, customtkinter; print('  [+] Flower Version : ' + str(flwr.__version__)); print('  [+] CustomTkinter  : ' + str(customtkinter.__version__))"

echo.
echo ======================================================================
echo [+] FedMedAI Environment Setup Complete!
echo [*] You can now launch the dashboard by running: run_dashboard.bat
echo ======================================================================
echo.
pause
