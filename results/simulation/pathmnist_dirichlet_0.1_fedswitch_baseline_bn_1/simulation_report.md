# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-09 14:52:11  
**Run Name:** `pathmnist_dirichlet_0.1_fedswitch_baseline_bn_1` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSWITCH** achieved **71.82%** accuracy in Round 66 (runtime: 719.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDSWITCH** | `dirichlet_0.1` | 69.97% | **71.82%** | **72.12%** (R62) | 1.0249 | **71.82%** (R66) | R66 | 1.2185 | 719.8s |