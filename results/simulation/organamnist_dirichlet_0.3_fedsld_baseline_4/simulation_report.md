# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-07 17:11:39  
**Run Name:** `organamnist_dirichlet_0.3_fedsld_baseline_4` | **Model:** `baseline` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSLD** achieved **84.74%** accuracy in Round 75 (runtime: 277.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | 571,707 | 2233.2 KB | `sgd` | No | **FEDSLD** | `dirichlet_0.3` | 84.27% | **84.74%** | **84.27%** (R75) | 0.6779 | **84.74%** (R75) | R75 | 0.6779 | 277.3s |