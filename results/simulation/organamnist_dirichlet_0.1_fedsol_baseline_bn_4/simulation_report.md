# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-09 11:21:51  
**Run Name:** `organamnist_dirichlet_0.1_fedsol_baseline_bn_4` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSOL** achieved **84.99%** accuracy in Round 79 (runtime: 372.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDSOL** | `dirichlet_0.1` | 84.59% | **84.99%** | **82.49%** (R79) | 0.6280 | **84.99%** (R79) | R79 | 0.5651 | 372.9s |