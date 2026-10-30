# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-28 15:14:16  
**Model:** `baseline` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] FEDAVG** achieved **77.80%** accuracy in Round 80 (runtime: 553.1s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | `sgd` | No | **FEDAVG** | `practical` | 77.80% | **77.80%** | **76.11%** (R30) | **77.80%** (R80) | R80 | 1.1390 | 553.1s |