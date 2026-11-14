# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 15:57:51  
**Run Name:** `bloodmnist_dirichlet_0.1_fedsol_baseline_bn_5` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSOL** achieved **84.95%** accuracy in Round 61 (runtime: 219.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDSOL** | `dirichlet_0.1` | 76.64% | **84.95%** | **82.01%** (R70) | 0.5697 | **84.95%** (R61) | R61 | 0.6693 | 219.9s |