# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 09:38:39  
**Run Name:** `organamnist_dirichlet_0.1_run_2_seed_43_baseline_bn_fedbn` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDBN** achieved **94.02%** accuracy in Round 80 (runtime: 294.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | `sgd` | No | **FEDBN** | `dirichlet_0.1` | 94.02% | **94.02%** | **94.18%** (R70) | **94.02%** (R80) | R80 | 0.2298 | 294.4s |