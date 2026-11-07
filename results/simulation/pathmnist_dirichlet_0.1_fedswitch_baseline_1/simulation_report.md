# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-07 12:57:12  
**Run Name:** `pathmnist_dirichlet_0.1_fedswitch_baseline_1` | **Model:** `baseline` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSWITCH** achieved **77.17%** accuracy in Round 59 (runtime: 696.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | 570,705 | 2229.3 KB | `sgd` | No | **FEDSWITCH** | `dirichlet_0.1` | 71.92% | **77.17%** | **77.74%** (R54) | 0.9453 | **77.17%** (R59) | R59 | 1.4846 | 696.6s |