# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 11:44:03  
**Run Name:** `bloodmnist_dirichlet_0.1_run_2_seed_43_baseline_bn_fedprox` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDPROX** achieved **83.31%** accuracy in Round 75 (runtime: 145.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | `sgd` | No | **FEDPROX** | `dirichlet_0.1` | 75.85% | **83.31%** | **85.48%** (R77) | **83.31%** (R75) | R75 | 0.7888 | 145.7s |