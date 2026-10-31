# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-31 08:33:50  
**Run Name:** `scr_bn_bloodmnist_s43` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = practical:** **[BEST] SCR** achieved **90.50%** accuracy in Round 76 (runtime: 1360.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **SCR** | `practical` | 90.41% | **90.50%** | **86.30%** (R76) | 0.4362 | **90.50%** (R76) | R76 | 0.2961 | 1360.8s |