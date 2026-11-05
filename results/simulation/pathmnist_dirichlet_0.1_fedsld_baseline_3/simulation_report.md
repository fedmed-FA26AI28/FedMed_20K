# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-05 21:29:32  
**Run Name:** `pathmnist_dirichlet_0.1_fedsld_baseline_3` | **Model:** `baseline` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSLD** achieved **72.48%** accuracy in Round 80 (runtime: 664.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | 570,705 | 2229.3 KB | `sgd` | No | **FEDSLD** | `dirichlet_0.1` | 72.48% | **72.48%** | **69.46%** (R21) | 1.3407 | **72.48%** (R80) | R80 | 1.0413 | 664.0s |