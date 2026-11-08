# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 09:50:47  
**Run Name:** `organamnist_dirichlet_0.1_run_2_seed_43_baseline_bn_proposed` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] PROPOSED** achieved **94.01%** accuracy in Round 78 (runtime: 307.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | `sgd` | No | **PROPOSED** | `dirichlet_0.1` | 93.75% | **94.01%** | **94.39%** (R73) | **94.01%** (R78) | R78 | 0.2310 | 307.5s |