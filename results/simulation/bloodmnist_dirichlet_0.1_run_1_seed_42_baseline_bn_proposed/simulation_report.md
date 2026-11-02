# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 11:38:53  
**Run Name:** `bloodmnist_dirichlet_0.1_run_1_seed_42_baseline_bn_proposed` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] PROPOSED** achieved **97.25%** accuracy in Round 78 (runtime: 110.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | `sgd` | No | **PROPOSED** | `dirichlet_0.1` | 97.25% | **97.25%** | **97.05%** (R72) | **97.25%** (R78) | R78 | 0.0967 | 110.2s |