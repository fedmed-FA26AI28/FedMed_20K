# FedMedAI: Heterogeneous Edge-Device Federated Learning for Medical Image Classification

FedMedAI is an enterprise-grade Federated Learning (FL) framework designed for medical imaging classification benchmarks under severe non-IID data distributions (Dirichlet, Pathological, and Practical cross-institution partitions). It provides seamless support for centralized baselines, distributed physical multi-device execution (PC server and NVIDIA Jetson Orin/Nano edge clients over LAN), and automated local multi-client simulation.

---

## 1. Key Features

- **Benchmark Datasets**: Native integration with MedMNIST datasets:
  - `pathmnist`: Colon pathology (9 classes, 3 channels RGB, 28x28)
  - `bloodmnist`: Peripheral blood cell types (8 classes, 3 channels RGB, 28x28)
  - `organamnist`: Abdominal CT organ patches (11 classes, 1-to-3 channels, 28x28)
  - `dermamnist`: Dermatoscopy pigmented skin lesions (7 classes, 3 channels RGB, 28x28)
- **Aggregation Strategies**:
  - `fedavg`: Vanilla Federated Averaging (McMahan et al., 2017)
  - `fedprox`: Proximal regularization for system heterogeneity (Li et al., 2020)
  - `fednova`: Normalized averaging for step-count heterogeneity (Wang et al., 2020)
  - `fedbn`: Local Batch Normalization for feature/domain shift (Li et al., 2021)
  - `scaffold`: Control variates for client drift correction (Karimireddy et al., 2020)
  - `proposed`: Synergistic integration of FedBN and SCAFFOLD (alias: `fedbn_scaffold`)
- **Non-IID Partition Schemes**:
  - `practical`: Shards per class modeling clinical institution size disparity (FedSLD benchmark). Fully generalized for arbitrary client counts $K \ge 1$ with exact 100% FedSLD reproduction when $K = 12$.
  - `pathological`: Subsets of classes assigned to clients, dynamically scaling to guarantee 100% class coverage across the federation without sample loss.
  - `dirichlet_X.X`: Continuous label distribution non-IID with concentration parameter $\alpha \in \{0.1, 0.3, 1.0\}$.
- **Dual Configuration System**:
  - `configs/experiment.yaml`: Dedicated solely to physical multi-device client execution over local network (PC and NVIDIA Jetson edge nodes).
  - `configs/simulation.yaml`: Dedicated solely to automated multi-client local simulation runs and matrix grids.
- **Hardware Telemetry & Profiling**: Automated monitoring of per-client round times, epoch throughput, ping latency, memory consumption, and straggler detection.

---

## 2. System Requirements

| Component | Specification |
| :--- | :--- |
| **Operating System** | Windows 10/11, Linux (Ubuntu 20.04/22.04 LTS), or JetPack (L4T) |
| **Python** | Python 3.8+ (Recommended: Python 3.10/3.11 virtual environment) |
| **Core Libraries** | `flwr>=1.7.0`, `torch>=2.0.0`, `torchvision`, `medmnist`, `pyyaml`, `scikit-learn`, `matplotlib`, `pandas`, `psutil` |
| **Hardware (Physical Deployment)** | 1 Central PC/Server + multiple physical clients (PC, Jetson Orin Nano, Jetson Nano 4GB) connected to the same LAN |

---

## 3. Directory Layout

```
FedMedAI_Study/
├── algorithms/                 # FL Strategy implementations
│   ├── base.py                 # Unified base FL strategy class with metrics aggregation
│   ├── fedavg.py               # Vanilla FedAvg
│   ├── fedprox.py              # FedProx strategy
│   ├── fednova.py              # FedNova strategy
│   ├── fedbn.py                # FedBN strategy
│   ├── scaffold.py             # SCAFFOLD strategy with control variates
│   ├── proposed.py             # Proposed Strategy (FedBN + SCAFFOLD)
│   └── coverage_la.py          # Coverage + Logit Adjustment (Coverage + LA)
├── client/                     # FL Client execution logic
│   ├── client.py               # Main Flower client implementation and CLI
│   ├── train.py                # Local training loop and optimizer handlers
│   ├── evaluate.py             # Local evaluation routines
│   └── system_monitor.py       # Hardware resource and latency telemetry
├── configs/                    # Configuration files
│   ├── experiment.yaml         # Real multi-device physical execution config
│   ├── simulation.yaml         # Automated simulation benchmark config
│   └── jetson.yaml             # Hardware-specific profiles (PC, Orin, Nano)
├── datasets/                   # Data loaders and partitioning
│   ├── medmnist_code.py        # MedMNIST loading, memory-mapping, and transforms
│   └── partition.py            # Non-IID partitioning algorithms and visualizers
├── experiments/                # Experiment scripts
│   ├── run_partition.py        # Standalone dataset partitioning CLI
│   ├── train_centralized.py    # Centralized upper-bound benchmark training
│   └── run_simulation.py       # Automated multi-client simulation harness
├── models/                     # Deep learning architectures
│   ├── cnn.py                  # Modern CNN (~20K parameters, BatchNorm)
│   ├── baseline.py             # Classic 4-layer CNN (~570K parameters, no BatchNorm)
│   └── factory.py              # Model instantiation factory
├── monitoring/                 # Metrics recording and dashboard utilities
│   ├── metrics.py              # FL metrics tracking and publication figures
│   └── dashboard.py            # Real-time monitoring dashboard
└── server/                     # FL Server execution
    └── server.py               # Flower server harness and evaluation hooks
```

---

## 4. Quickstart Guide

### 4.1. Prerequisites & Environment Setup

Activate the virtual environment:
- **Windows PowerShell**:
  ```powershell
  D:\FPT\KLTN\FED\.venv\Scripts\Activate.ps1
  ```
- **Linux / Jetson Bash**:
  ```bash
  source .venv/bin/activate
  ```

### 4.2. Centralized Baseline Training

Train a centralized upper-bound model on the entire dataset:
```bash
python -m experiments.train_centralized --dataset pathmnist --model cnn --epochs 100 --batch_size 256 --lr 0.01
```

### 4.3. Data Partitioning

Generate non-IID partition splits with paired local test sets:
```bash
# Practical partition on PathMNIST for 12 clients
python -m experiments.run_partition --dataset pathmnist --partition_mode practical --num_clients 12 --local_test

# Pathological partition on DermaMNIST for 5 clients
python -m experiments.run_partition --dataset dermamnist --partition_mode pathological --num_clients 5 --local_test
```

### 4.4. Real Multi-Device Deployment (Physical Clients over LAN)

1. **Start Server** on the central machine:
   ```bash
   python -m server.server --config configs/experiment.yaml
   ```
2. **Start Clients** on edge devices:
   ```bash
   # Client 0 (e.g., PC client)
   python -m client.client --config configs/experiment.yaml --client_id 0 --server_address 192.168.1.100:8080

   # Client 1 (e.g., Jetson Orin edge node)
   python -m client.client --config configs/experiment.yaml --client_id 1 --server_address 192.168.1.100:8080 --device_type jetson_orin
   ```

### 4.5. Automated Multi-Client Simulation

Run an automated simulation across strategies and partitions on a single machine:
```bash
python -m experiments.run_simulation --config configs/simulation.yaml
```

For complete CLI options and detailed step-by-step guides, refer to [run.md](run.md).
