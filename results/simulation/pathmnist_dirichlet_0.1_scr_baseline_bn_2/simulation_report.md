# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-04 18:10:49  
**Run Name:** `dirichlet01_bn_pathmnist_s43` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] SCR** achieved **67.86%** accuracy in Round 67 (runtime: 4216.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **SCR** | `dirichlet_0.1` | 61.27% | **67.86%** | **69.54%** (R67) | 1.3052 | **67.86%** (R67) | R67 | 1.5915 | 4216.9s |