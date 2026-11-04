# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-04 08:57:24  
**Run Name:** `dirichlet01_bn_bloodmnist_s46` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] SCR** achieved **89.83%** accuracy in Round 76 (runtime: 666.1s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDAVG** | `dirichlet_0.1` | 82.99% | **87.93%** | **85.01%** (R72) | 0.4126 | **87.93%** (R75) | R75 | 0.4718 | 674.9s |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **SCR** | `dirichlet_0.1` | 89.10% | **89.83%** | **88.06%** (R76) | 0.3606 | **89.83%** (R76) | R76 | 0.3035 | 666.1s |