# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-07 15:11:44  
**Run Name:** `organamnist_dirichlet_0.1_fedsld_baseline_1` | **Model:** `baseline` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSLD** achieved **82.76%** accuracy in Round 80 (runtime: 309.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | 571,707 | 2233.2 KB | `sgd` | No | **FEDSLD** | `dirichlet_0.1` | 78.95% | **82.76%** | **78.95%** (R79) | 0.6754 | **82.76%** (R80) | R80 | 0.6754 | 309.5s |