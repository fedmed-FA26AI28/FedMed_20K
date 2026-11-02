# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 13:42:15  
**Run Name:** `bloodmnist_dirichlet_0.3_fedsld_baseline_bn_2` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSLD** achieved **90.85%** accuracy in Round 76 (runtime: 102.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDSLD** | `dirichlet_0.3` | 86.88% | **90.85%** | **91.82%** (R76) | 0.2301 | **90.85%** (R76) | R76 | 0.3422 | 102.8s |