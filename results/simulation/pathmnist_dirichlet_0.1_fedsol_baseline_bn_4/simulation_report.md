# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-09 16:42:45  
**Run Name:** `pathmnist_dirichlet_0.1_fedsol_baseline_bn_4` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSOL** achieved **67.28%** accuracy in Round 79 (runtime: 932.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDSOL** | `dirichlet_0.1` | 63.16% | **67.28%** | **65.92%** (R71) | 1.7582 | **67.28%** (R79) | R79 | 1.3913 | 932.5s |