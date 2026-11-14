# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 09:57:32  
**Run Name:** `bloodmnist_dirichlet_0.1_fedsol_baseline_2` | **Model:** `baseline` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSOL** achieved **87.34%** accuracy in Round 75 (runtime: 208.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDSOL** | `dirichlet_0.1` | 86.99% | **87.34%** | **87.70%** (R75) | 0.3952 | **87.34%** (R75) | R75 | 0.4182 | 208.5s |