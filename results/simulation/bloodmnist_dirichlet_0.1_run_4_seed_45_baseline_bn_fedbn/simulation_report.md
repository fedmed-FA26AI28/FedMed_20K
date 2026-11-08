# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 12:08:12  
**Run Name:** `bloodmnist_dirichlet_0.1_run_4_seed_45_baseline_bn_fedbn` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDBN** achieved **96.29%** accuracy in Round 76 (runtime: 93.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | `sgd` | No | **FEDBN** | `dirichlet_0.1` | 95.85% | **96.29%** | **93.15%** (R76) | **96.29%** (R76) | R76 | 0.1521 | 93.6s |