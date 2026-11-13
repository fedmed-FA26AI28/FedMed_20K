# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-13 23:19:31  
**Run Name:** `bloodmnist_dirichlet_0.1_fedswitch_baseline_1` | **Model:** `baseline` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSWITCH** achieved **89.07%** accuracy in Round 78 (runtime: 164.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDSWITCH** | `dirichlet_0.1` | 88.80% | **89.07%** | **89.62%** (R66) | 0.3229 | **89.07%** (R78) | R78 | 0.3109 | 164.2s |