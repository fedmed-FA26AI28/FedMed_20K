# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 11:46:17  
**Run Name:** `bloodmnist_dirichlet_0.1_run_2_seed_43_baseline_bn_fedbn` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDBN** achieved **97.54%** accuracy in Round 80 (runtime: 122.1s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | `sgd` | No | **FEDBN** | `dirichlet_0.1` | 97.54% | **97.54%** | **96.20%** (R80) | **97.54%** (R80) | R80 | 0.0940 | 122.1s |