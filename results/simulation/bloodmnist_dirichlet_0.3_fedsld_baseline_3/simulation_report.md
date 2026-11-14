# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 10:46:54  
**Run Name:** `bloodmnist_dirichlet_0.3_fedsld_baseline_3` | **Model:** `baseline` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSLD** achieved **86.76%** accuracy in Round 76 (runtime: 159.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDSLD** | `dirichlet_0.3` | 86.83% | **86.76%** | **86.83%** (R77) | 0.3774 | **86.76%** (R76) | R76 | 0.3774 | 159.7s |