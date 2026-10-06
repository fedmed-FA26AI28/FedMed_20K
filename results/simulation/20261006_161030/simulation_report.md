# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-06 16:20:54  
**Rounds:** 20 | **Clients:** 10 | **Local Epochs:** 5 | **LR Mode:** `server_decay`  

---

## 1. Executive Summary & Best Performing Strategies

- **Alpha = 1.0 (Near-IID):** 🏆 **FEDBN** achieved **77.66%** accuracy in Round 19 (total runtime: 226.1s).
- **Alpha = 0.3 (Moderate Non-IID):** 🏆 **FEDBN** achieved **65.04%** accuracy in Round 19 (total runtime: 250.5s).
- **Alpha = 0.1 (High Non-IID):** 🏆 **FEDBN** achieved **23.74%** accuracy in Round 3 (total runtime: 111.8s).

---

## 2. Full Simulation Results Matrix

| Strategy | Alpha | Final Acc (%) | Best Acc (%) | Best Round | Final Loss | Best Loss | Total Time | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FEDBN** | `1.0` | 77.20% | **77.66%** | Round 19 | 0.6430 | 0.6339 | 226.1s | `completed` |
| **FEDBN** | `0.3` | 64.53% | **65.04%** | Round 19 | 0.9643 | 0.9531 | 250.5s | `completed` |
| **FEDBN** | `0.1` | 20.25% | **23.74%** | Round 3 | 3.3613 | 3.0300 | 111.8s | `completed` |

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
