# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 09:45:27  
**Run Name:** `organamnist_dirichlet_0.1_run_2_seed_43_baseline_bn_scaffold` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] SCAFFOLD** achieved **84.49%** accuracy in Round 71 (runtime: 395.1s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | `sgd` | No | **SCAFFOLD** | `dirichlet_0.1` | 84.33% | **84.49%** | **84.55%** (R76) | **84.49%** (R71) | R71 | 0.5198 | 395.1s |