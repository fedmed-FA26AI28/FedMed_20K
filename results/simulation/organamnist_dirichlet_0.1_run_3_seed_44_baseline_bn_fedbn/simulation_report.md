# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 10:07:46  
**Run Name:** `organamnist_dirichlet_0.1_run_3_seed_44_baseline_bn_fedbn` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDBN** achieved **94.79%** accuracy in Round 77 (runtime: 263.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | `sgd` | No | **FEDBN** | `dirichlet_0.1` | 94.79% | **94.79%** | **94.28%** (R77) | **94.79%** (R77) | R77 | 0.1959 | 263.9s |