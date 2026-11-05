# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-05 16:19:17  
**Run Name:** `pathmnist_dirichlet_0.1_scaffold_baseline_bn_1` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] SCAFFOLD** achieved **77.70%** accuracy in Round 76 (runtime: 678.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **SCAFFOLD** | `dirichlet_0.1` | 69.55% | **77.70%** | **78.66%** (R76) | 0.8145 | **77.70%** (R76) | R76 | 1.0237 | 678.2s |