# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-05 15:32:54  
**Run Name:** `pathmnist_dirichlet_0.1_fedprox_baseline_bn_2` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDPROX** achieved **65.95%** accuracy in Round 61 (runtime: 716.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDPROX** | `dirichlet_0.1` | 59.37% | **65.95%** | **67.85%** (R46) | 1.2043 | **65.95%** (R61) | R61 | 1.3300 | 716.4s |