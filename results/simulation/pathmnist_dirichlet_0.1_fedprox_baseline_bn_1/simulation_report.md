# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-05 15:20:41  
**Run Name:** `pathmnist_dirichlet_0.1_fedprox_baseline_bn_1` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDPROX** achieved **70.29%** accuracy in Round 74 (runtime: 712.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDPROX** | `dirichlet_0.1` | 68.77% | **70.29%** | **72.07%** (R74) | 1.0987 | **70.29%** (R74) | R74 | 1.1268 | 712.2s |