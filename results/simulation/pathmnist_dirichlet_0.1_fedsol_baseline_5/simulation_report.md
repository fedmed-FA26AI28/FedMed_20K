# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-08 13:20:13  
**Run Name:** `pathmnist_dirichlet_0.1_fedsol_baseline_5` | **Model:** `baseline` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSOL** achieved **76.87%** accuracy in Round 57 (runtime: 781.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | 570,705 | 2229.3 KB | `sgd` | No | **FEDSOL** | `dirichlet_0.1` | 72.69% | **76.87%** | **78.14%** (R32) | 1.0843 | **76.87%** (R57) | R57 | 1.6222 | 781.7s |