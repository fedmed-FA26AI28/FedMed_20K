# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-07 15:34:33  
**Run Name:** `organamnist_dirichlet_0.1_fedsld_baseline_5` | **Model:** `baseline` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSLD** achieved **83.21%** accuracy in Round 79 (runtime: 389.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | 571,707 | 2233.2 KB | `sgd` | No | **FEDSLD** | `dirichlet_0.1` | 84.39% | **83.21%** | **84.39%** (R80) | 0.6814 | **83.21%** (R79) | R79 | 0.6814 | 389.4s |