# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-11 00:41:41  
**Run Name:** `dirichlet03_bn_bloodmnist_s43` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] SCR** achieved **92.43%** accuracy in Round 78 (runtime: 635.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **SCR** | `dirichlet_0.3` | 92.82% | **92.43%** | **92.82%** (R69) | 0.2170 | **92.43%** (R78) | R78 | 0.2170 | 635.7s |