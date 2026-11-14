# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 15:46:51  
**Run Name:** `bloodmnist_dirichlet_0.1_fedsol_baseline_bn_2` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSOL** achieved **75.15%** accuracy in Round 75 (runtime: 222.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDSOL** | `dirichlet_0.1` | 74.22% | **75.15%** | **76.88%** (R80) | 0.6288 | **75.15%** (R75) | R75 | 0.7064 | 222.7s |