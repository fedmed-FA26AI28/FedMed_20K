# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 16:15:50  
**Run Name:** `bloodmnist_dirichlet_0.3_fedprox_baseline_bn_1` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDPROX** achieved **93.19%** accuracy in Round 77 (runtime: 161.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDPROX** | `dirichlet_0.3` | 93.77% | **93.19%** | **93.77%** (R77) | 0.1927 | **93.19%** (R77) | R77 | 0.1927 | 161.9s |