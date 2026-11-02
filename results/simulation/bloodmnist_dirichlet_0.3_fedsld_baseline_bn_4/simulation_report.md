# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 13:45:57  
**Run Name:** `bloodmnist_dirichlet_0.3_fedsld_baseline_bn_4` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSLD** achieved **92.05%** accuracy in Round 74 (runtime: 100.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDSLD** | `dirichlet_0.3` | 90.18% | **92.05%** | **92.24%** (R74) | 0.2300 | **92.05%** (R74) | R74 | 0.2636 | 100.2s |