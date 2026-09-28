# Session Summary: 1e00e435-a97b-4406-89b1-b6c55f26e374

- **Date**: 2026-09-28
- **Topic**: Updated `.gitignore`, added comprehensive Linux/Jetson command guide to `run.md`, and trimmed `requirements.txt` to active dependencies with unpinned PyTorch
- **Active Branch**: `DuyKhang1`

## Key Accomplishments & Changes

1. **Updated `.gitignore`**:
   - Added Python virtual environment patterns (`.venv*`, `env.bak/`, `venv.bak/`).
   - Added build and compilation artifacts (`dist/`, `build/`, `*.so`, `*.egg-info/`, etc.).
   - Added checkpoints and model weight formats (`*.onnx`, `*.safetensors`, `*.ckpt`, `checkpoints/`).
   - Added raw/downloaded dataset archives (`data/raw/`, `data/medmnist/`, `*.tar.gz`, `*.zip`).
   - Added experiment tracking and Jetson hardware monitoring logs (`monitoring/logs/`, `tegrastats*.log`, `jetson_stats*.log`, `wandb/`, `runs/`, `lightning_logs/`).
   - Added test/linter caches (`.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.coverage`).
   - Added environment variable files (`.env*`, preserving `!.env.example`).
   - Added IDE and notebook checkpoints (`.idea/`, `.vscode/`, `.ipynb_checkpoints/`).
   - Added agent scratch and session log rules (`agent_gen/`, `memories/**/.system_generated/`, `memories/**/*.jsonl`, `memories/**/output.txt`).
   - Kept repository status clean and uncluttered.

2. **Added Comprehensive Linux & Jetson Command Guide to `run.md`**:
   - **Environment Setup**: Added instructions for Linux venv creation, pip upgrade, requirements installation, and JetPack ARM64 wheel verification on Jetson.
   - **Model Verification**: Added `python3 -m models.cnn` for Linux.
   - **Data Partitioning**: Added Linux bash commands for `--all`, specific configurations (`--num_clients 10 --alpha 0.3 --seed 42`), and quick test runs (`--num_clients 3`).
   - **Centralized Baseline**: Added direct execution and background execution with `nohup` (`nohup python3 -m experiments.train_centralized > centralized_training.log 2>&1 &`).
   - **Data Distribution**: Added SCP distribution command and individual file transfer examples.
   - **Local Simulation**:
     - Added server launch commands across `fedavg`, `fedprox`, `fednova`, and `--server_eval`.
     - Added auto-spawn bash loop to start 3 clients in parallel in the background with separate log files, plus termination via `pkill -f "client.client"`.
   - **Real LAN Cluster Deployment**:
     - Server setup: finding host IP (`hostname -I`), opening firewall port 8080 (`sudo ufw allow 8080/tcp`).
     - Jetson Hardware Configuration: setting maximum power mode and clocks (`sudo nvpmodel -m 0`, `sudo jetson_clocks`) on Jetson Orin and Nano.
     - Jetson Hardware Monitoring: logging resources via `tegrastats` and interactive inspection with `jtop`.
     - Detailed manual client startup commands for all 5 Jetson Orin (0-4) and 5 Jetson Nano (5-9).
     - Automated Cluster Script: provided `start_cluster.sh` and `stop_cluster.sh` for one-click remote SSH execution across all 10 Jetson boards.
   - **Session Persistence**: Added `tmux` cheatsheet for long-running FL training on Linux.
   - **Parameters Reference**: Maintained and verified parameter lookup table.

3. **Cleaned `requirements.txt`**:
   - Conducted AST code audit across all files in repository (`algorithms/`, `client/`, `datasets/`, `experiments/`, `models/`, `monitoring/`, `scripts/`, `server/`).
   - Filtered down from 88 bloated/transitive packages to only the 10 libraries actually imported in the codebase:
     - `torch` (version requirement removed for JetPack/CUDA compatibility)
     - `torchvision` (version requirement removed)
     - `flwr>=1.15.0`
     - `medmnist>=3.0.0`
     - `numpy`
     - `scikit-learn>=1.0.0`
     - `matplotlib>=3.5.0`
     - `seaborn>=0.12.0`
     - `psutil>=5.8.0`
     - `PyYAML>=6.0`
   - Removed all unused web/database dependencies (`fastapi`, `uvicorn`, `SQLAlchemy`, `alembic`, `ray`, etc.) and personal git repository links.
   - Verified dependency graph with `pip check` (no broken requirements).

4. **Made Centralized Baseline Configurable (`experiments/train_centralized.py`)**:
   - Analyzed `experiments/train_centralized.py` and resolved the user's question regarding whether epochs were hardcoded.
   - Replaced hard-coded parameters (`epochs = 100`, `batch_size = 32`, `learning_rate = 0.001`) with CLI argument parsing via `argparse`.
   - Maintained 100% backward compatibility by defaulting to `epochs=100`, `batch_size=32`, and `lr=0.001`.
   - Added Windows console character encoding protection (`sys.stdout.reconfigure`) against cp932/cp1252 `UnicodeEncodeError`.
   - Recorded hyperparameters inside `centralized_results.json`.
   - Updated documentation in `run.md` with usage examples and parameter reference.

