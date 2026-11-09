# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-09 12:18:37  
**Run Name:** `organamnist_dirichlet_0.3_fedsol_baseline_bn_4` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSOL** achieved **87.01%** accuracy in Round 58 (runtime: 366.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDSOL** | `dirichlet_0.3` | 82.87% | **87.01%** | **87.28%** (R58) | 0.4359 | **87.01%** (R58) | R58 | 0.5914 | 366.6s |