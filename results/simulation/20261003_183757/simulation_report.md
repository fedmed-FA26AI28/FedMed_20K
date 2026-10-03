# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-03 19:41:46  
**Rounds:** 20 | **Clients:** 10 | **Local Epochs:** 5 | **LR Mode:** `server_decay`  

---

## 1. Executive Summary & Best Performing Strategies

- **Alpha = 1.0 (Near-IID):** 🏆 **SCAFFOLD** achieved **87.11%** accuracy in Round 19 (total runtime: 275.3s).
- **Alpha = 0.3 (Moderate Non-IID):** 🏆 **SCAFFOLD** achieved **84.24%** accuracy in Round 19 (total runtime: 308.9s).
- **Alpha = 0.1 (High Non-IID):** 🏆 **FEDPROX** achieved **68.20%** accuracy in Round 9 (total runtime: 211.9s).

---

## 2. Full Simulation Results Matrix

| Strategy | Alpha | Final Acc (%) | Best Acc (%) | Best Round | Final Loss | Best Loss | Total Time | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FEDAVG** | `1.0` | 84.92% | **85.06%** | Round 15 | 0.4868 | 0.4951 | 245.2s | `completed` |
| **FEDPROX** | `1.0` | 85.47% | **85.56%** | Round 17 | 0.4972 | 0.4999 | 282.3s | `completed` |
| **FEDNOVA** | `1.0` | 86.58% | **86.58%** | Round 20 | 0.4790 | 0.4790 | 244.3s | `completed` |
| **FEDBN** | `1.0` | 67.17% | **67.17%** | Round 7 | 0.9039 | 0.9039 | 95.8s | `completed` |
| **SCAFFOLD** | `1.0` | 86.90% | **87.11%** | Round 19 | 0.4489 | 0.4496 | 275.3s | `completed` |
| **FEDAVG** | `0.3` | 82.84% | **83.43%** | Round 18 | 0.5506 | 0.5545 | 334.3s | `completed` |
| **FEDPROX** | `0.3` | 81.47% | **82.43%** | Round 18 | 0.5805 | 0.5864 | 393.3s | `completed` |
| **FEDNOVA** | `0.3` | 82.64% | **82.64%** | Round 19 | 0.6246 | 0.6236 | 329.2s | `completed` |
| **FEDBN** | `0.3` | 43.44% | **45.48%** | Round 4 | 2.4066 | 2.1262 | 105.0s | `completed` |
| **SCAFFOLD** | `0.3` | 84.01% | **84.24%** | Round 19 | 0.5852 | 0.5850 | 308.9s | `completed` |
| **FEDAVG** | `0.1` | 64.81% | **66.59%** | Round 11 | 1.1753 | 1.2298 | 216.3s | `completed` |
| **FEDPROX** | `0.1` | 63.64% | **68.20%** | Round 9 | 1.2138 | 1.2510 | 211.9s | `completed` |
| **FEDNOVA** | `0.1` | 63.20% | **63.20%** | Round 20 | 1.2472 | 1.2472 | 251.4s | `completed` |
| **FEDBN** | `0.1` | 26.05% | **26.05%** | Round 7 | 2.8576 | 2.8576 | 96.8s | `completed` |
| **SCAFFOLD** | `0.1` | 59.95% | **65.36%** | Round 12 | 1.1178 | 1.1310 | 241.5s | `completed` |

---

## 3. Generated Comparison Artifacts

- 📊 **Accuracy Curves:** `compare_accuracy_curves.png`
- 📉 **Loss Curves:** `compare_loss_curves.png`
- 📶 **Accuracy Bar Chart:** `compare_accuracy_barchart.png`
- ⏱️ **Runtime Comparison:** `compare_runtime_barchart.png`
- 🗺️ **Accuracy Heatmap:** `compare_accuracy_heatmap.png`
- 📋 **Aggregated Summary CSV:** `simulation_summary.csv`
- 📜 **Round History CSV:** `simulation_rounds_history.csv`
- 💾 **Machine-Readable JSON:** `simulation_results.json`
