# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-13 21:06:36  
**Run Name:** `bloodmnist_practical_fedavg_baseline_1` | **Model:** `baseline` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = practical:** **[BEST] FEDAVG** achieved **88.25%** accuracy in Round 80 (runtime: 375.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDAVG** | `practical` | 84.93% | **88.25%** | **84.93%** (R80) | 0.4165 | **88.25%** (R80) | R80 | 0.4165 | 375.4s |