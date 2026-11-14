# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 17:50:09  
**Run Name:** `bloodmnist_dirichlet_0.3_fedsol_baseline_bn_2` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSOL** achieved **87.61%** accuracy in Round 58 (runtime: 195.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDSOL** | `dirichlet_0.3` | 83.48% | **87.61%** | **88.97%** (R58) | 0.3372 | **87.61%** (R58) | R58 | 0.5462 | 195.4s |