# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 13:56:29  
**Run Name:** `organamnist_dirichlet_0.3_fedsld_baseline_bn_2` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSLD** achieved **86.78%** accuracy in Round 75 (runtime: 219.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDSLD** | `dirichlet_0.3` | 85.66% | **86.78%** | **86.94%** (R79) | 0.4347 | **86.78%** (R75) | R75 | 0.4460 | 219.8s |