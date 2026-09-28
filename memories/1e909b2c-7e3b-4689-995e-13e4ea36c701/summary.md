# Session Summary: 1e909b2c-7e3b-4689-995e-13e4ea36c701

- **Date**: 2026-09-28
- **Topic**: Verified Multi-PC LAN compatibility for Federated Learning and restructured `run.md` into 3 designated sections with options references.
- **Active Branch**: `DuyKhang1`

## Key Accomplishments & Changes

1. **Verified Multi-PC LAN Compatibility for Federated Learning**:
   - Confirmed that Flower utilizes gRPC over standard TCP sockets, making it fully cross-platform and hardware-agnostic (Windows, Linux, Jetson, macOS).
   - Confirmed `server/server.py` binds to `0.0.0.0:8080` by default to listen on all network interfaces across LAN.
   - Confirmed `client/client.py` accepts `--server_address <IP>:<PORT>` to connect to any LAN host.
   - PyTorch automatically selects CUDA on GPU PCs or falls back to CPU if no discrete GPU is available.
   - Documented firewall considerations (`New-NetFirewallRule` for Windows, `ufw allow 8080/tcp` for Linux).

2. **Restructured `run.md` into 3 Dedicated Sections with CLI Options Notes**:
   - **Section 1: Centralized run on PC**:
     - Execution commands for Windows PowerShell and Linux Bash.
     - Default execution vs custom parameters vs background `nohup` execution.
     - Dedicated options reference note covering `--epochs`, `--batch_size`, `--lr` / `--learning_rate`, explanation, defaults, and examples.
   - **Section 2: FL run on multiple PC**:
     - Architecture diagram of Server PC + multiple Client PCs across LAN.
     - Partition generation & distribution prerequisites.
     - Server setup (finding LAN IP via `ipconfig`/`hostname -I`, firewall configuration, running FedAvg, FedProx, FedNova).
     - Client setup (running Client ID 0, 1, 2 on respective PCs with `--server_address`).
     - Dedicated options reference note covering Server options (`--strategy`, `--rounds`, `--min_clients`, `--local_epochs`, `--learning_rate`, `--proximal_mu`, `--early_stop_patience`, `--server_eval`, `--host`, `--port`) and Client options (`--client_id`, `--server_address`, `--num_clients`, `--alpha`, `--partition_path`, `--local_epochs`, `--learning_rate`, `--batch_size`).
   - **Section 3: FL run on Jetson device with Linux**:
     - Architecture diagram of Server + 5 Jetson Orin + 5 Jetson Nano cluster.
     - Automated and manual SCP partition data distribution.
     - Hardware power mode and clock configuration (`sudo nvpmodel -m 0`, `sudo jetson_clocks`).
     - Hardware resource monitoring via `tegrastats` and `jtop`.
     - Server startup and manual client execution across Orin (0-4) and Nano (5-9).
     - Automated remote cluster execution scripts (`start_cluster.sh`, `stop_cluster.sh`).
     - Dedicated options reference note covering automatic hardware configuration mapping from `configs/jetson.yaml` (Orin: batch 32, 2 workers; Nano: batch 16, 0 workers), client CLI arguments, and system management tools (`nvpmodel`, `jetson_clocks`, `tegrastats`, `jtop`, `pkill`, `tmux`).
