# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-07 15:26:01  
**Run Name:** `pathmnist_dirichlet_0.1_fedsol_baseline_bn_1` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSOL** achieved **72.13%** accuracy in Round 45 (runtime: 987.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDSOL** | `dirichlet_0.1` | 63.25% | **72.13%** | **69.59%** (R59) | 1.1921 | **72.13%** (R45) | R45 | 1.2233 | 987.3s |