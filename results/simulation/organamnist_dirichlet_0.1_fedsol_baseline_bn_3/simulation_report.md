# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-07 13:30:52  
**Run Name:** `organamnist_dirichlet_0.1_fedsol_baseline_bn_3` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSOL** achieved **84.62%** accuracy in Round 71 (runtime: 394.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDSOL** | `dirichlet_0.1` | 84.52% | **84.62%** | **84.65%** (R71) | 0.4728 | **84.62%** (R71) | R71 | 0.4873 | 394.9s |