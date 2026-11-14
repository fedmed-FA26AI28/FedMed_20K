# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 15:23:18  
**Run Name:** `bloodmnist_dirichlet_0.1_fedsld_baseline_bn_5` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSLD** achieved **84.83%** accuracy in Round 60 (runtime: 197.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDSLD** | `dirichlet_0.1` | 79.71% | **84.83%** | **79.71%** (R60) | 0.5308 | **84.83%** (R60) | R60 | 0.5308 | 197.2s |