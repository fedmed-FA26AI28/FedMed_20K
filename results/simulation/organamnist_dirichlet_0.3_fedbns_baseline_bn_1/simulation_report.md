# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-04 10:54:33  
**Run Name:** `organamnist_dirichlet_0.3_fedbns_baseline_bn_1` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDBNS** achieved **91.14%** accuracy in Round 79 (runtime: 296.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDBNS** | `dirichlet_0.3` | 90.97% | **91.14%** | **91.15%** (R79) | 0.3750 | **91.14%** (R79) | R79 | 0.3795 | 296.6s |