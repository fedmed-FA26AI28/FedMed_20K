# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 13:52:36  
**Run Name:** `organamnist_dirichlet_0.3_fedsld_baseline_bn_1` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSLD** achieved **87.28%** accuracy in Round 80 (runtime: 270.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDSLD** | `dirichlet_0.3` | 87.28% | **87.28%** | **87.13%** (R75) | 0.4404 | **87.28%** (R80) | R80 | 0.4333 | 270.0s |