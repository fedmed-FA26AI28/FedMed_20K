# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-28 14:36:42  
**Model:** `baseline` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = practical:** **[BEST] FEDAVG** achieved **87.52%** accuracy in Round 77 (runtime: 136.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | `sgd` | No | **FEDAVG** | `practical` | 87.52% | **87.52%** | **86.61%** (R72) | **87.52%** (R77) | R77 | 0.3469 | 136.3s |