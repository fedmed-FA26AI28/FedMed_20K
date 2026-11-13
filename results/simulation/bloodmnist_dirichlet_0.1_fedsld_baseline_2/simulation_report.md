# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-13 23:06:42  
**Run Name:** `bloodmnist_dirichlet_0.1_fedsld_baseline_2` | **Model:** `baseline` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSLD** achieved **80.30%** accuracy in Round 74 (runtime: 199.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDSLD** | `dirichlet_0.1` | 80.36% | **80.30%** | **80.36%** (R77) | 0.5203 | **80.30%** (R74) | R74 | 0.5203 | 199.9s |