# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-28 14:34:15  
**Model:** `baseline` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = practical:** **[BEST] FEDAVG** achieved **85.41%** accuracy in Round 78 (runtime: 122.1s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | `sgd` | No | **FEDAVG** | `practical` | 85.38% | **85.41%** | **79.81%** (R78) | **85.41%** (R78) | R78 | 0.4052 | 122.1s |