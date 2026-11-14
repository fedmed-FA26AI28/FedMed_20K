# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 16:19:01  
**Run Name:** `bloodmnist_dirichlet_0.3_fedprox_baseline_bn_2` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDPROX** achieved **92.28%** accuracy in Round 79 (runtime: 176.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDPROX** | `dirichlet_0.3` | 92.75% | **92.28%** | **92.75%** (R79) | 0.2015 | **92.28%** (R79) | R79 | 0.2015 | 176.0s |