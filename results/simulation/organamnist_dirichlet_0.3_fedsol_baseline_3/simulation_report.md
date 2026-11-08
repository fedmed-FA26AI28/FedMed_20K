# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-08 10:23:36  
**Run Name:** `organamnist_dirichlet_0.3_fedsol_baseline_3` | **Model:** `baseline` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSOL** achieved **85.00%** accuracy in Round 80 (runtime: 319.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | 571,707 | 2233.2 KB | `sgd` | No | **FEDSOL** | `dirichlet_0.3` | 85.00% | **85.00%** | **84.53%** (R79) | 0.6067 | **85.00%** (R80) | R80 | 0.5794 | 319.5s |