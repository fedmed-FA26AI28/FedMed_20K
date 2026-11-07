# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-07 15:22:52  
**Run Name:** `organamnist_dirichlet_0.1_fedsld_baseline_3` | **Model:** `baseline` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSLD** achieved **81.25%** accuracy in Round 77 (runtime: 307.1s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | 571,707 | 2233.2 KB | `sgd` | No | **FEDSLD** | `dirichlet_0.1` | 81.33% | **81.25%** | **81.33%** (R79) | 0.6751 | **81.25%** (R77) | R77 | 0.6751 | 307.1s |