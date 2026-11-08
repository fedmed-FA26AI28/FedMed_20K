# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 09:26:42  
**Run Name:** `organamnist_dirichlet_0.1_run_2_seed_43_baseline_bn_fedavg` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDAVG** achieved **84.32%** accuracy in Round 79 (runtime: 379.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | `sgd` | No | **FEDAVG** | `dirichlet_0.1` | 83.33% | **84.32%** | **84.08%** (R64) | **84.32%** (R79) | R79 | 0.5400 | 379.4s |