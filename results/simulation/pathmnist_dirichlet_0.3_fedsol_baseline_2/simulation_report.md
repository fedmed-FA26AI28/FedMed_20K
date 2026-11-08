# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-08 14:26:01  
**Run Name:** `pathmnist_dirichlet_0.3_fedsol_baseline_2` | **Model:** `baseline` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSOL** achieved **80.45%** accuracy in Round 65 (runtime: 741.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | 570,705 | 2229.3 KB | `sgd` | No | **FEDSOL** | `dirichlet_0.3` | 79.93% | **80.45%** | **78.82%** (R77) | 1.4628 | **80.45%** (R65) | R65 | 1.4163 | 741.6s |