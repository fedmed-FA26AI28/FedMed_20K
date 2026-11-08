# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 09:20:10  
**Run Name:** `organamnist_dirichlet_0.1_run_1_seed_42_baseline_bn_proposed` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] PROPOSED** achieved **94.09%** accuracy in Round 76 (runtime: 304.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | `sgd` | No | **PROPOSED** | `dirichlet_0.1` | 92.93% | **94.09%** | **93.44%** (R76) | **94.09%** (R76) | R76 | 0.2655 | 304.4s |