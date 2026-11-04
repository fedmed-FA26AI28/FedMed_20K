# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-04 07:19:37  
**Run Name:** `dirichlet01_bn_bloodmnist_s42` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDAVG** achieved **92.02%** accuracy in Round 73 (runtime: 610.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDAVG** | `dirichlet_0.1` | 87.14% | **92.02%** | **92.71%** (R73) | 0.2555 | **92.02%** (R73) | R73 | 0.3435 | 610.9s |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **SCR** | `dirichlet_0.1` | 89.33% | **90.12%** | **90.97%** (R67) | 0.2813 | **90.12%** (R67) | R67 | 0.3221 | 618.2s |