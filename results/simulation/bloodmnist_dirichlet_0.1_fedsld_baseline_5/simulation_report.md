# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-13 23:16:33  
**Run Name:** `bloodmnist_dirichlet_0.1_fedsld_baseline_5` | **Model:** `baseline` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSLD** achieved **83.37%** accuracy in Round 80 (runtime: 193.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDSLD** | `dirichlet_0.1` | 79.56% | **83.37%** | **79.56%** (R76) | 0.5514 | **83.37%** (R80) | R80 | 0.5514 | 193.7s |