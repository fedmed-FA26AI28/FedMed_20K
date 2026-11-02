# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 14:01:00  
**Run Name:** `organamnist_dirichlet_0.3_fedsld_baseline_bn_3` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSLD** achieved **86.73%** accuracy in Round 69 (runtime: 256.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDSLD** | `dirichlet_0.3` | 86.44% | **86.73%** | **87.64%** (R77) | 0.4656 | **86.73%** (R69) | R69 | 0.5024 | 256.8s |