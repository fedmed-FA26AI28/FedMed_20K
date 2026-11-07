# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-07 17:37:07  
**Run Name:** `pathmnist_dirichlet_0.3_fedsld_baseline_2` | **Model:** `baseline` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSLD** achieved **77.67%** accuracy in Round 63 (runtime: 621.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | 570,705 | 2229.3 KB | `sgd` | No | **FEDSLD** | `dirichlet_0.3` | 76.26% | **77.67%** | **76.26%** (R63) | 1.1351 | **77.67%** (R63) | R63 | 1.1351 | 621.9s |