# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-08 10:13:00  
**Run Name:** `organamnist_dirichlet_0.3_fedsol_baseline_1` | **Model:** `baseline` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSOL** achieved **85.45%** accuracy in Round 79 (runtime: 340.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | 571,707 | 2233.2 KB | `sgd` | No | **FEDSOL** | `dirichlet_0.3` | 85.36% | **85.45%** | **84.64%** (R70) | 0.6101 | **85.45%** (R79) | R79 | 0.5937 | 340.6s |