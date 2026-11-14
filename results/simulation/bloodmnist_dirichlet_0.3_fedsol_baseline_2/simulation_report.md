# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 11:13:37  
**Run Name:** `bloodmnist_dirichlet_0.3_fedsol_baseline_2` | **Model:** `baseline` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSOL** achieved **88.28%** accuracy in Round 75 (runtime: 183.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDSOL** | `dirichlet_0.3` | 81.64% | **88.28%** | **89.28%** (R75) | 0.3243 | **88.28%** (R75) | R75 | 0.5496 | 183.9s |