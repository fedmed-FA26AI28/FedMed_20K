# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 15:16:42  
**Run Name:** `bloodmnist_dirichlet_0.1_fedsld_baseline_bn_3` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSLD** achieved **82.49%** accuracy in Round 57 (runtime: 175.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDSLD** | `dirichlet_0.1` | 83.57% | **82.49%** | **83.57%** (R57) | 0.5093 | **82.49%** (R57) | R57 | 0.5093 | 175.6s |