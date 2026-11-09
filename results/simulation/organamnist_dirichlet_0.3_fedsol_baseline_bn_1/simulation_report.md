# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-09 12:00:10  
**Run Name:** `organamnist_dirichlet_0.3_fedsol_baseline_bn_1` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSOL** achieved **87.84%** accuracy in Round 75 (runtime: 401.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDSOL** | `dirichlet_0.3` | 87.75% | **87.84%** | **87.34%** (R78) | 0.5082 | **87.84%** (R75) | R75 | 0.4974 | 401.6s |