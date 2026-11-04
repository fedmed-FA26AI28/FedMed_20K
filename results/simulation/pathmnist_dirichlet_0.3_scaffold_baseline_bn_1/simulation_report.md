# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-04 11:27:22  
**Run Name:** `pathmnist_dirichlet_0.3_scaffold_baseline_bn_1` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] SCAFFOLD** achieved **84.53%** accuracy in Round 76 (runtime: 855.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **SCAFFOLD** | `dirichlet_0.3` | 84.01% | **84.53%** | **84.30%** (R76) | 0.9114 | **84.53%** (R76) | R76 | 0.9025 | 855.9s |