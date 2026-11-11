@echo off
title FedMedAI Federated Orchestrator ^& Live Monitor
cd /d "%~dp0"

echo ======================================================================
echo        FedMedAI Federated Learning Orchestrator ^& Live Monitor
echo ======================================================================
echo.

:: 1. Prioritize virtual environment inside the project root
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
    goto :RUN
)

:: 2. Fallback to parent virtual environment if present
if exist "..\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=..\.venv\Scripts\python.exe"
    goto :RUN
)

:: 3. If no venv found, automatically trigger setup.bat
echo [!] Virtual environment not detected.
echo [*] Triggering setup.bat to configure Python and PyTorch GPU automatically...
echo.
call setup.bat
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
    goto :RUN
)

:: Fallback to system python
set "PYTHON_EXE=python"

:RUN
echo [*] Launching dashboard with: %PYTHON_EXE%
echo.
"%PYTHON_EXE%" run_dashboard.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [!] Dashboard exited with error code %ERRORLEVEL%.
    pause
)
