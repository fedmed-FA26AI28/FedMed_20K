# Session Summary: 17076fc3-34f0-41d1-8e65-2eec392096ff

- **Date**: 2026-09-27
- **Topic**: Repository Overview & Usage Guide (How to use this repo)
- **Active Branch**: `DuyKhang1`

## Key Accomplishments & Repository State
1. **Audited repository structure & progress against Implementation Plan**:
   - **Phase 3 (Centralized Baseline)**: Fully implemented in `experiments/train_centralized.py`, utilizing `models/cnn.py` (SimpleCNN), `client/train.py`, and `client/evaluate.py`. Tested with multiple historical runs stored in `results/centralized/`.
   - **Phase 4 (Non-IID Partitioning)**: Implemented in `datasets/partition.py` and executable via `experiments/run_partition.py` with Dirichlet α values (1.0, 0.3, 0.1) across 3, 5, 10 clients. Partitions and plots are saved in `data/partitions/`.
   - **Phase 5+ (FL Implementation)**: `client/client.py`, `server/server.py`, `algorithms/fedavg.py`, `algorithms/proposed.py`, and `monitoring/` are placeholder stubs awaiting implementation.
   - **Deployment Utilities**: `scripts/distribute_data.py` ready for automated SCP distribution of client partitions to 5 Jetson Orin + 5 Jetson Nano edge boards.
2. **Provided Clear Usage Instructions**:
   - Environment setup and requirements installation.
   - Verifying the model architecture.
   - Generating Dirichlet data partitions.
   - Running the centralized baseline training and viewing summary cards/curves.
   - Distributing data partitions to Jetson edge devices.
   - Next steps for building out the Federated Learning client/server loop.

## Follow-up Update
- User provided path to active venv: `D:\FPT\KLTN\FED\.venv`.
- Verified environment:
  - Flower version: `1.30.0`
  - PyTorch version: `2.14.0+cu130` (CUDA available: `True`)
  - MedMNIST version: `3.0.2`
- Codebase status for Flower:
  - Prerequisite components (`models/cnn.py` parameter serialization, `client/train.py`, `client/evaluate.py`, `datasets/partition.py`) are ready.
  - Entry points `client/client.py` and `server/server.py` are stubs that need Flower `NumPyClient` and `start_server` implementation.

## Model Replacement
- Updated `models/cnn.py`:
  - Replaced architecture with requested `CNN` (ConvBlock1: 3->32, ConvBlock2: 32->64, MaxPool2d, AdaptiveAvgPool2d, Linear: 64->num_classes).
  - Added `count_parameters(model)`.
  - Preserved `get_parameters` and `set_parameters` for Flower compatibility.
  - Added `SimpleCNN = CNN` alias for backwards compatibility.
  - Validated architecture with `models.cnn`: Total parameters = 20,104; Trainable parameters = 20,104.
  - Updated `experiments/train_centralized.py` to instantiate `CNN`.

## Federated Learning Strategies & Full Pipeline Implementation
- Implemented FL strategies in `algorithms/`:
  - `algorithms/fedavg.py`: `FedAvgStrategy` wrapping Flower `FedAvg`, custom metrics aggregator for client metrics, per-round timing, server-side early stopping, and run summary.
  - `algorithms/fedprox.py`: `FedProxStrategy` with proximal mu regularizer support, timing, early stopping, and weight-size logging.
  - `algorithms/fednova.py`: `FedNovaStrategy` normalized averaging addressing objective inconsistency across heterogeneous client steps.
  - `algorithms/__init__.py`: Factory `get_strategy(name="fedavg", **kwargs)` and registry.
  - `algorithms/proposed.py`: Re-exports `ProposedStrategy` (FedProx) and `FedNovaStrategy`.
- Updated `client/client.py`:
  - Flower `FedMedAIClient(fl.client.NumPyClient)`.
  - Local training loop with learning rate scheduling (`ReduceLROnPlateau`) and early stopping.
  - Supports FedProx proximal term (`(proximal_mu / 2.0) * ||w - w_global||^2`).
  - Logs and returns: `epoch_time_avg`, `training_time`, `weight_size_kb` (size of weights sent to server), `local_steps`.
- Updated `server/server.py`:
  - Uses `from algorithms import get_strategy` (imports from local algorithms directory, not directly from flwr).
  - CLI argument `--strategy` with default `"fedavg"`, selectable between `fedavg`, `fedprox`, `fednova`.
  - Tracks and logs: total time of each round (`round_time_seconds`), total time for all rounds (`total_elapsed_seconds`), client epoch time stats, straggler latency, weight payload stats.
  - Server-side early stopping based on validation/test accuracy across rounds.
  - Saves final results to `results/federated/<strategy>/<timestamp>/fl_results.json`.
- All imports and class instantiations verified with `task-171`.

## Documentation & Runbook
- Created `run.md` with complete command line instructions covering:
  - Virtual environment activation (`D:\FPT\KLTN\FED\.venv`).
  - Model verification (`models/cnn.py`).
  - Dirichlet non-IID partition creation (`experiments/run_partition.py`).
  - Centralized baseline training (`experiments/train_centralized.py`).
  - SCP data distribution to Jetson devices (`scripts/distribute_data.py`).
  - Federated learning local simulation (Server + multi-client).
  - Federated learning live cluster deployment (Server + 5 Jetson Orin + 5 Jetson Nano).
  - CLI arguments reference table for both server and client.
