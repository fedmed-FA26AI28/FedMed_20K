# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-04 07:43:13  
**Run Name:** `dirichlet01_bn_bloodmnist_s43` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDAVG** achieved **89.54%** accuracy in Round 75 (runtime: 680.1s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDAVG** | `dirichlet_0.1` | 84.71% | **89.54%** | **89.65%** (R75) | 0.2844 | **89.54%** (R75) | R75 | 0.3973 | 680.1s |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **SCR** | `dirichlet_0.1` | 76.73% | **85.65%** | **85.58%** (R75) | 0.3673 | **85.65%** (R75) | R75 | 0.9325 | 692.4s |