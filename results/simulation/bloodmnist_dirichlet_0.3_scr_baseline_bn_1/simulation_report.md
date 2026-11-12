# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-11 00:30:39  
**Run Name:** `dirichlet03_bn_bloodmnist_s42` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] SCR** achieved **92.78%** accuracy in Round 77 (runtime: 628.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **SCR** | `dirichlet_0.3` | 93.17% | **92.78%** | **93.17%** (R77) | 0.2142 | **92.78%** (R77) | R77 | 0.2142 | 628.4s |