# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-22 18:22:19  
**Model:** `cnn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `server_decay` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] FEDAVG** achieved **83.11%** accuracy in Round 47 (runtime: 6490.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `cnn` | `adam` | Yes | **FEDAVG** | `practical` | 77.02% | **83.11%** | **79.61%** (R47) | **83.11%** (R47) | R47 | 0.7554 | 6490.2s |