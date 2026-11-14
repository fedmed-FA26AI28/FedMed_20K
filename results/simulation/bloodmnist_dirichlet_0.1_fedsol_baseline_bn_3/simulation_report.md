# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 15:50:27  
**Run Name:** `bloodmnist_dirichlet_0.1_fedsol_baseline_bn_3` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSOL** achieved **86.26%** accuracy in Round 77 (runtime: 201.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDSOL** | `dirichlet_0.1` | 84.83% | **86.26%** | **87.42%** (R77) | 0.3885 | **86.26%** (R77) | R77 | 0.4426 | 201.0s |