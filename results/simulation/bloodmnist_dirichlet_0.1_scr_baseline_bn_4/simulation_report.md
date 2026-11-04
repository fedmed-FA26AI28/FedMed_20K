# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-04 08:34:20  
**Run Name:** `dirichlet01_bn_bloodmnist_s45` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDAVG** achieved **87.34%** accuracy in Round 70 (runtime: 596.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDAVG** | `dirichlet_0.1` | 80.97% | **87.34%** | **79.75%** (R78) | 0.5314 | **87.34%** (R70) | R70 | 0.4962 | 596.3s |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **SCR** | `dirichlet_0.1` | 82.72% | **87.14%** | **83.80%** (R49) | 0.4333 | **87.14%** (R49) | R49 | 0.5193 | 608.2s |