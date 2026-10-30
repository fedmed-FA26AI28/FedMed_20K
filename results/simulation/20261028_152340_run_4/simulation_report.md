# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-28 15:35:01  
**Model:** `baseline` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] FEDAVG** achieved **80.97%** accuracy in Round 66 (runtime: 669.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | `sgd` | No | **FEDAVG** | `practical` | 80.17% | **80.97%** | **79.71%** (R48) | **80.97%** (R66) | R66 | 0.9449 | 669.3s |