# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 10:58:33  
**Run Name:** `bloodmnist_dirichlet_0.3_fedswitch_baseline_2` | **Model:** `baseline` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSWITCH** achieved **90.21%** accuracy in Round 79 (runtime: 160.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDSWITCH** | `dirichlet_0.3` | 89.86% | **90.21%** | **90.79%** (R79) | 0.2733 | **90.21%** (R79) | R79 | 0.2902 | 160.6s |