# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-13 23:03:07  
**Run Name:** `bloodmnist_dirichlet_0.1_fedsld_baseline_1` | **Model:** `baseline` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSLD** achieved **83.51%** accuracy in Round 79 (runtime: 177.1s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDSLD** | `dirichlet_0.1` | 86.55% | **83.51%** | **86.55%** (R79) | 0.3865 | **83.51%** (R79) | R79 | 0.3865 | 177.1s |