# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-11 01:03:46  
**Run Name:** `dirichlet03_bn_bloodmnist_s45` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] SCR** achieved **92.69%** accuracy in Round 60 (runtime: 645.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **SCR** | `dirichlet_0.3` | 92.77% | **92.69%** | **92.77%** (R60) | 0.2366 | **92.69%** (R60) | R60 | 0.2366 | 645.4s |