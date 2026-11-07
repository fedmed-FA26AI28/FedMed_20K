# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-07 15:27:46  
**Run Name:** `organamnist_dirichlet_0.1_fedsld_baseline_4` | **Model:** `baseline` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSLD** achieved **81.71%** accuracy in Round 80 (runtime: 278.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | 571,707 | 2233.2 KB | `sgd` | No | **FEDSLD** | `dirichlet_0.1` | 79.37% | **81.71%** | **79.37%** (R73) | 0.7236 | **81.71%** (R80) | R80 | 0.7236 | 278.9s |