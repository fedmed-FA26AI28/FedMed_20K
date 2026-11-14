# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 11:20:07  
**Run Name:** `bloodmnist_dirichlet_0.3_fedsol_baseline_4` | **Model:** `baseline` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSOL** achieved **90.50%** accuracy in Round 76 (runtime: 182.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDSOL** | `dirichlet_0.3` | 88.13% | **90.50%** | **90.02%** (R76) | 0.3221 | **90.50%** (R76) | R76 | 0.3540 | 182.7s |