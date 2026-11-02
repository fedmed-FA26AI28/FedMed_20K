# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 11:34:28  
**Run Name:** `bloodmnist_dirichlet_0.1_run_1_seed_42_baseline_bn_fedbn` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDBN** achieved **96.99%** accuracy in Round 79 (runtime: 99.1s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | `sgd` | No | **FEDBN** | `dirichlet_0.1` | 96.99% | **96.99%** | **96.80%** (R69) | **96.99%** (R79) | R79 | 0.0980 | 99.1s |