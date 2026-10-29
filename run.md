# FedMedAI: Execution and Run Guide

This document provides complete instructions for executing the **FedMedAI** system across three primary operational modes:
1. **Centralized Baseline Training on PC**: Train CNN models on the full dataset without federation to establish the empirical upper-bound benchmark.
2. **Distributed FL on Multiple Physical Devices (PC & Jetson Cluster)**: Run Federated Learning across real machines connected over local network (LAN) using TCP/gRPC sockets.
3. **Automated Multi-Strategy Simulation Benchmark**: Run automated multi-client FL simulations locally on a single machine across partition schemes and strategies.

---

## 0. Prerequisites and Environment Setup

Activate the project Python virtual environment:

- **Windows PowerShell**:
  ```powershell
  D:\FPT\KLTN\FED\.venv\Scripts\Activate.ps1
  ```
- **Linux (Ubuntu PC / Jetson)**:
  ```bash
  source .venv/bin/activate
  ```

Verify neural network architectures:
- **Modern CNN** (~20K parameters with BatchNorm and AdaptiveAvgPool):
  ```bash
  python -m models.cnn
  ```
- **Baseline CNN** (~570K parameters, classic 4-layer CNN without BatchNorm matching FedSLD ISBI 2022):
  ```bash
  python -m models.baseline
  ```

---

## 1. Centralized Baseline Training on PC

Centralized training optimizes the model on the full training set without partitioning, establishing the empirical upper bound for comparison against federated strategies.

### 1.1. Execution Examples

- **Default Execution** (100 epochs, batch size 256, initial lr 0.01):
  ```bash
  python -m experiments.train_centralized --config configs/experiment.yaml
  ```
- **Custom Hyperparameters** (e.g. 50 epochs on PathMNIST with Adam optimizer):
  ```bash
  python -m experiments.train_centralized --dataset pathmnist --model cnn --optimizer adam --epochs 50 --batch_size 64 --lr 0.001
  ```
- **Linux Background Execution (`nohup`)**:
  ```bash
  nohup python3 -m experiments.train_centralized --dataset pathmnist --epochs 100 --batch_size 256 > centralized.log 2>&1 &
  tail -f centralized.log
  ```

### 1.2. Output Artifacts

Artifacts are automatically saved to `results/centralized/<timestamp>/` (or `results/centralized/<name>/` when `--name` is provided):
- `best_model.pth`: Model weights achieving the highest validation accuracy.
- `training_curves.png`: Loss and accuracy curves across training and validation epochs.
- `confusion_matrix.png`: Test set confusion matrix.
- `per_class_metrics.png`: Per-class precision, recall, and F1-score chart.
- `centralized_results.json`: Complete metrics report and hyperparameters.
- `summary_card.png`: Visual executive summary card of the experiment.

### 1.3. CLI Options Reference for `experiments/train_centralized.py`

| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--config` | `str` | `configs/experiment.yaml` | Path to YAML configuration file containing centralized settings (`centralized:` block). |
| `--name`, `-n` | `str` | `None` | Designated run identifier for the output folder name (`results/centralized/<name>/`). |
| `--output_dir`, `--save_dir` | `str` | `None` | Custom output directory to store centralized training artifacts. |
| `--dataset` | `str` | `pathmnist` | Dataset name: `bloodmnist`, `pathmnist`, `organamnist`, `dermamnist`. |
| `--model` | `str` | `cnn` | Architecture: `cnn` (Modern CNN with BatchNorm) or `baseline` (FedSLD classic CNN). |
| `--optimizer` | `str` | `adam` | Optimizer type: `adam` or `sgd`. |
| `--momentum` | `float` | `0.0` | Momentum factor when using `--optimizer sgd`. |
| `--epochs` | `int` | `100` | Maximum training epochs (early stopping automatically halts when validation loss plateaus). |
| `--batch_size` | `int` | `256` | DataLoader mini-batch size. |
| `--lr`, `--learning_rate` | `float` | `0.01` | Initial learning rate. |
| `--lr_patience` | `int` | `5` | Epochs without validation loss improvement before `ReduceLROnPlateau` decays LR. |
| `--lr_factor` | `float` | `0.5` | Multiplicative factor of learning rate reduction (`new_lr = lr * factor`). |
| `--min_lr` | `float` | `1e-6` | Lower bound for learning rate. |
| `--early_stop_patience` | `int` | `15` | Epochs without validation loss improvement before stopping training early. |
| `--augment` / `--no-augment` | `bool` | `True` | Toggle training data augmentation (Random Crop and Horizontal Flip). |
| `--random_crop` / `--no-random_crop` | `bool` | `True` | Toggle RandomCrop (padding=2, reflection mode) on 28x28 images. |
| `--random_flip` / `--no-random_flip` | `bool` | `True` | Toggle RandomHorizontalFlip on training images. |
| `--seed` | `int` | `42` | Global random seed for reproducible initialization and data shuffling. |

---

## 2. Distributed FL on Multiple Physical Devices (PC & Jetson Cluster)

This operational mode deploys Federated Learning across multiple physical devices communicating over local network (LAN) via TCP/gRPC sockets.

```
       ┌────────────────────────────────────────────────────────┐
       │   FL Server (Central PC / Ubuntu Server)               │
       │   IP Address: 192.168.1.100 : 8080                     │
       └───────────┬────────────────────────────────┬───────────┘
                   │                                │
                   ▼ (LAN TCP/gRPC)                 ▼
      ┌─────────────────────────┐      ┌─────────────────────────┐
      │   Client 0 (PC Node)    │      │   Client 1 (Jetson Orin)│
      │   IP: 192.168.1.101     │      │   IP: 192.168.1.102     │
      └─────────────────────────┘      └─────────────────────────┘
```

### 2.1. Data Partitioning

Generate non-IID partition splits with paired local test sets on the server (or locally on each client machine):

- **Dirichlet Non-IID** ($\alpha = 0.3$, 3 clients):
  ```bash
  python -m experiments.run_partition --num_clients 3 --dataset bloodmnist --partition_mode dirichlet_0.3 --seed 42 --local_test
  ```
- **Practical Non-IID** (Simulating institution size disparity, 12 clients):
  ```bash
  python -m experiments.run_partition --num_clients 12 --dataset pathmnist --partition_mode practical --seed 42 --local_test
  ```
- **Pathological Non-IID** (Random class assignment, 12 clients):
  ```bash
  python -m experiments.run_partition --num_clients 12 --dataset dermamnist --partition_mode pathological --seed 42 --local_test
  ```

Partition artifacts are saved to `data/partitions/`:
- `partition_<dataset>_<mode>_localtest_seed<seed>_clients<num>.json`: Contains `client_indices` for training and `test_indices` for local client evaluation.
- `data/partitions/plots/dist_<dataset>_<mode>_localtest_seed<seed>_clients<num>.png`: Visual distribution comparison chart.

### 2.2. Server Execution (`server/server.py`)

1. Verify server LAN IP address (e.g. `192.168.1.100` via `ipconfig` on Windows or `hostname -I` on Linux).
2. Allow inbound traffic on port 8080:
   - **Windows (PowerShell as Admin)**:
     ```powershell
     New-NetFirewallRule -DisplayName "Flower FL Server 8080" -Direction Inbound -LocalPort 8080 -Protocol TCP -Action Allow
     ```
   - **Linux**:
     ```bash
     sudo ufw allow 8080/tcp
     ```
3. Launch FL Server using `configs/experiment.yaml`:
   ```bash
   python -m server.server --config configs/experiment.yaml
   ```
   Or override parameters directly via CLI:
   ```bash
   # Launch Proposed Strategy (FedBN + SCAFFOLD) for 12 clients over 40 rounds:
   python -m server.server --strategy proposed --model cnn --rounds 40 --min_clients 12 --local_test --host 0.0.0.0 --port 8080
   ```

### 2.3. Client Execution (`client/client.py`)

Run client instances on each physical machine pointing to the server address:

- **Client 0 (PC Node)**:
  ```bash
  python -m client.client --config configs/experiment.yaml --client_id 0 --server_address 192.168.1.100:8080 --device_type pc
  ```
- **Client 1 (NVIDIA Jetson Orin)**:
  ```bash
  python3 -m client.client --config configs/experiment.yaml --client_id 1 --server_address 192.168.1.100:8080 --device_type jetson_orin
  ```
- **Client 2 (NVIDIA Jetson Nano)**:
  ```bash
  python3 -m client.client --config configs/experiment.yaml --client_id 2 --server_address 192.168.1.100:8080 --device_type jetson_nano
  ```

### 2.4. CLI Options Reference for Server & Client

#### Server Options (`server/server.py`)
| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--config` | `str` | `configs/experiment.yaml` | Path to YAML experiment configuration file. |
| `--strategy` | `str` | `proposed` | Aggregation strategy: `fedavg`, `fedprox`, `fednova`, `fedbn`, `scaffold`, `fedbns`, `proposed`, `cla`, `scr`. |
| `--coverage_kappa` | `float` | `512.0` | CLA coverage anchoring parameter kappa (controls weight decay curve). |
| `--coverage_mu_head` | `float` | `0.1` | CLA classifier head penalty intensity mu_head (anchors to global head). |
| `--la_tau` | `float` | `1.0` | CLA Logit Adjustment intensity tau (shifts logits by class frequency). |
| `--la_epsilon` | `float` | `1.0` | CLA Logit Adjustment smoothing epsilon for log counts. |
| `--dataset` | `str` | `pathmnist` | Dataset name: `bloodmnist`, `pathmnist`, `organamnist`, `dermamnist`. |
| `--model` | `str` | `cnn` | Model architecture: `cnn` or `baseline`. |
| `--rounds` | `int` | `40` | Total communication rounds. |
| `--min_clients` | `int` | `12` | Minimum connected clients required before each round starts. |
| `--local_epochs` | `int` | `5` | Local training epochs per client per round. |
| `--lr_mode` | `str` | `fixed` | Learning rate mode: `fixed`, `client_loss`, or `server_decay`. |
| `--learning_rate` | `float` | `0.01` | Initial base learning rate. |
| `--local_test` | `bool` | `True` | Track BMCTA and BTA on client-specific local test sets (required for FedBN and Proposed). |
| `--server_eval` | `bool` | `True` | Evaluate aggregated model on server global test set each round. |
| `--proximal_mu` | `float` | `0.01` | FedProx proximal penalty coefficient. |
| `--early_stop_patience`| `int` | `0` | Rounds without metric improvement before stopping (0 = disabled). |
| `--host` | `str` | `0.0.0.0` | Socket bind address. |
| `--port` | `int` | `8080` | Socket TCP port. |

#### Client Options (`client/client.py`)
| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--client_id` | `int` | *(Required)* | Unique client identifier integer (0, 1, 2, ...). |
| `--server_address` | `str` | `127.0.0.1:8080` | Server LAN address in `<IP>:<PORT>` format. |
| `--config` | `str` | `configs/experiment.yaml` | Path to YAML experiment configuration file. |
| `--dataset` | `str` | `pathmnist` | Dataset name: `bloodmnist`, `pathmnist`, `organamnist`, `dermamnist`. |
| `--partition_mode` | `str` | `practical` | Partition mode (`practical`, `pathological`, `dirichlet_0.3`, etc.). |
| `--num_clients` | `int` | `12` | Total number of clients in partition split. |
| `--strategy` | `str` | `fedbns` | Client FL strategy (`fedavg`, `fedprox`, `fednova`, `fedbn`, `scaffold`, `fedbns`, `cla`, etc.). |
| `--optimizer` | `str` | `sgd` | Client local optimizer: `adam`, `sgd`, or `adamw`. |
| `--coverage_kappa` | `float` | `512.0` | CLA coverage anchoring parameter kappa. |
| `--coverage_mu_head` | `float` | `0.1` | CLA classifier head penalty intensity mu_head. |
| `--la_tau` | `float` | `1.0` | CLA Logit Adjustment intensity tau. |
| `--la_epsilon` | `float` | `1.0` | CLA Logit Adjustment smoothing epsilon. |
| `--device_type` | `str` | `pc` | Hardware profile identifier: `pc`, `jetson_orin`, `jetson_nano`. |
| `--local_test` | `bool` | `True` | Load paired local test split for client evaluation. |
| `--batch_size` | `int` | `256` | DataLoader mini-batch size. |
| `--num_workers` | `int` | `0` | DataLoader worker threads (0 is recommended for network clients). |
| `--save_metrics` | `bool` | `True` | Save local CSV metrics and latency logs. |

---

## 3. NVIDIA Jetson Edge Cluster Deployment

For physical deployments on embedded NVIDIA Jetson hardware (Jetson AGX Orin, Orin Nano, and Jetson Nano):

### 3.1. Power Modes and Clock Locking

Lock hardware clocks to maximum frequencies before benchmarking to ensure consistent execution latency:

- **NVIDIA Jetson Orin**:
  ```bash
  sudo nvpmodel -m 0          # MAXN power mode
  sudo jetson_clocks          # Lock CPU and GPU to maximum frequencies
  ```
- **NVIDIA Jetson Nano**:
  ```bash
  sudo nvpmodel -m 0          # 10W MAX power mode
  sudo jetson_clocks          # Lock clocks
  ```

### 3.2. Hardware Telemetry Monitoring

Monitor hardware resource utilization and power draw during training:
```bash
# Log hardware telemetry every 1 second:
sudo tegrastats --interval 1000 --logfile ~/FedMedAI/tegrastats_client.log &

# Live interactive dashboard:
jtop
```

### 3.3. Remote Cluster Automation Scripts

Manage multiple edge devices simultaneously via SSH:

- **Start Cluster (`scripts/start_cluster.sh`)**:
  ```bash
  #!/bin/bash
  SERVER="192.168.1.100:8080"
  DEVICES=("192.168.1.101" "192.168.1.102" "192.168.1.103")
  for i in "${!DEVICES[@]}"; do
    IP=${DEVICES[$i]}
    echo "Starting Client $i on $IP..."
    ssh -n jetson@$IP "cd ~/FedMedAI && nohup python3 -m client.client --config configs/experiment.yaml --client_id $i --server_address $SERVER > ~/FedMedAI/client_$i.log 2>&1 &"
  done
  ```
- **Stop Cluster (`scripts/stop_cluster.sh`)**:
  ```bash
  #!/bin/bash
  DEVICES=("192.168.1.101" "192.168.1.102" "192.168.1.103")
  for IP in "${DEVICES[@]}"; do
    ssh -n jetson@$IP "pkill -f 'client.client'"
  done
  echo "All client processes terminated."
  ```

---

## 4. Automated Multi-Strategy Simulation Benchmark

The simulation harness ([`experiments/run_simulation.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/experiments/run_simulation.py)) enables automated, multi-run benchmarking locally on a single machine across multiple datasets, non-IID partitions, and FL strategies using [`configs/simulation.yaml`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/configs/simulation.yaml).

### 4.1. Supported Benchmark Strategies

| Strategy | Local BN (FedBN) | Control Variates (SCAFFOLD) | Addressed Challenge |
| :--- | :---: | :---: | :--- |
| `fedavg` | No | No | Baseline without heterogeneity compensation. |
| `fedprox` | No | No | Proximal regularization restricts client drift. |
| `fednova` | No | No | Normalized averaging resolves local step-count heterogeneity. |
| `fedbn` | Yes | No | Local Batch Normalization mitigates feature/domain shift. |
| `scaffold` | No | Yes | Control variates correct client drift from label non-IID. |
| `fedbns` (`proposed`) | Yes | Yes | **Proposed Strategy**: Synergistic integration of FedBN and SCAFFOLD. |
| `cla` | No | No | **Coverage + Logit Adjustment (CoverageLA)**: Local classifier head anchoring penalty to global reference + class-frequency logit offset. |

### 4.2. Running Simulations

- **Execute Default Simulation Grid**:
  ```bash
  python -m experiments.run_simulation --config configs/simulation.yaml
  ```
- **Single Fast Verification Run** (1 round, 2 clients, FedAvg):
  ```bash
  python -m experiments.run_simulation --rounds 1 --local_epochs 1 --num_clients 2 --strategies fedavg --partitions dirichlet_1.0
  ```
- **Paper Baseline Replication** (FedSLD Table 2: 12 clients, Practical partition, SGD lr=0.01, batch 256, 80 rounds):
  ```bash
  python -m experiments.run_simulation --datasets pathmnist --partitions practical --strategies fedavg --model baseline --optimizer sgd --learning_rate 0.01 --lr_mode fixed --batch_size 256 --rounds 80 --local_test
  ```
- **CLA (Coverage + Logit Adjustment) Benchmark Run** (80 rounds, PathMNIST, baseline architecture):
  ```bash
  python -m experiments.run_simulation --datasets pathmnist --partitions practical --strategies cla --model baseline --optimizer sgd --rounds 80 --local_test
  ```
- **Full Benchmark Suite on DermaMNIST**:
  ```bash
  python -m experiments.run_simulation --datasets dermamnist --partitions practical pathological --strategies fedavg fedprox fednova fedbn scaffold proposed --model cnn --rounds 40 --num_clients 12 --local_test
  ```

### 4.3. Simulation Output Structure

Simulation outputs are compiled in `results/simulation/<name_or_timestamp>/`:
- `simulation_summary.csv`: Aggregated accuracy, BMCTA, BTA, loss, and runtime metrics for all runs.
- `simulation_rounds_history.csv`: Round-by-round trajectory across all benchmarked configurations.
- `simulation_results.json`: Complete machine-readable metadata and results.
- `simulation_report.md`: Markdown summary report with leaderboard rankings.
- `<dataset>_compare_accuracy_curves.png`: Cross-strategy accuracy convergence curves.
- `<dataset>_compare_loss_curves.png`: Cross-strategy loss convergence curves.
- `<dataset>_compare_accuracy_barchart.png`: Comparison bar chart of final and peak test accuracy.
- `<dataset>_compare_runtime_barchart.png`: Runtime comparison bar chart.
- `<dataset>/<partition>/<strategy>/`: Individual run artifacts (models, CSVs, parameter dynamics).

### 4.4. CLI Options Reference for Simulation (`experiments/run_simulation.py`)

| CLI Option | Config Key (`configs/simulation.yaml`) | Default | Description |
| :--- | :--- | :---: | :--- |
| `--config` | *(file path)* | `configs/simulation.yaml` | Path to YAML simulation config. |
| `--name`, `-n` | `name` | `None` | Custom output directory name (`results/simulation/<name>/`). |
| `--datasets` | `datasets` | `[pathmnist]` | List of datasets: `bloodmnist`, `pathmnist`, `organamnist`, `dermamnist`. |
| `--partitions` | `partitions` | `[practical]` | List of partitions: `practical`, `pathological`, `dirichlet_0.3`, `dirichlet_0.1`, `dirichlet_1.0`. |
| `--strategies` | `strategies` | `[proposed]` | List of strategies: `fedavg`, `fedprox`, `fednova`, `fedbn`, `scaffold`, `proposed`. |
| `--num_clients` | `num_clients` | `12` | Number of simulated clients participating in the federation. |
| `--rounds` | `rounds` | `40` | Total communication rounds per run. |
| `--local_epochs` | `local_epochs` | `5` | Local epochs per client per round. |
| `--batch_size` | `batch_size` | `256` | DataLoader mini-batch size. |
| `--model` | `model` | `cnn` | Architecture: `cnn` or `baseline`. |
| `--optimizer` | `optimizer` | `adam` | Optimizer type: `adam` or `sgd`. |
| `--learning_rate`, `--lr` | `learning_rate` | `0.01` | Base initial learning rate. |
| `--lr_mode` | `lr_mode` | `fixed` | LR scheduling mode: `fixed`, `client_loss`, `server_decay`. |
| `--local_test` | `local_test` | `True` | Partition test set across clients and record BMCTA / BTA. |
| `--server_eval` | `server_eval` | `True` | Evaluate global test set at server. |
| `--proximal_mu` | `proximal_mu` | `0.01` | Proximal penalty $\mu$ for FedProx. |
| `--coverage_kappa` | `coverage_kappa` | `512.0` | CLA coverage anchoring parameter kappa ($\kappa > 0$). |
| `--coverage_mu_head` | `coverage_mu_head` | `0.1` | CLA classifier head penalty intensity $\mu_{\text{head}} \ge 0$. |
| `--la_tau` | `la_tau` | `1.0` | CLA Logit Adjustment intensity $\tau \ge 0$. |
| `--la_epsilon` | `la_epsilon` | `1.0` | CLA Logit Adjustment smoothing $\epsilon > 0$. |
| `--client_module` | `client_module` | `client.client` | Subprocess client engine (`client.client` for parallel; `experiments.simulation_client` for GPU lock). |
| `--client_cpu_threads` | `client_cpu_threads` | `1` | PyTorch CPU threads per subprocess (prevents CPU thread thrashing). |
| `--timeout_per_run` | `timeout_per_run` | `0` | Maximum timeout seconds per run (<=0 disables timeout). |
