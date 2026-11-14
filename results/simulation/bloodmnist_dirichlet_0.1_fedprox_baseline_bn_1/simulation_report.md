# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 14:04:28  
**Run Name:** `bloodmnist_dirichlet_0.1_fedprox_baseline_bn_1` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDPROX** achieved **90.65%** accuracy in Round 80 (runtime: 176.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDPROX** | `dirichlet_0.1` | 90.94% | **90.65%** | **90.94%** (R80) | 0.2701 | **90.65%** (R80) | R80 | 0.2701 | 176.7s |