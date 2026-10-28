# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-28 21:10:58  
**Run Name:** `bloodmnist_fedavg_baseline_bn_2` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = practical:** **[BEST] FEDAVG** achieved **89.89%** accuracy in Round 80 (runtime: 115.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDAVG** | `practical` | 89.89% | **89.89%** | **86.23%** (R80) | 0.3724 | **89.89%** (R80) | R80 | 0.2803 | 115.5s |