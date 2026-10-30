# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-23 16:51:05  
**Run Name:** `fedavg_sd` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `server_decay` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] FEDAVG** achieved **81.03%** accuracy in Round 58 (runtime: 1740.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | `sgd` | Yes | **FEDAVG** | `practical` | 79.92% | **81.03%** | **78.64%** (R70) | **81.03%** (R58) | R58 | 0.9874 | 1740.3s |