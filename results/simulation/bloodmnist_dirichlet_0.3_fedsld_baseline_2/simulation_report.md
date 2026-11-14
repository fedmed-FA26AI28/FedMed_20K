# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 10:43:59  
**Run Name:** `bloodmnist_dirichlet_0.3_fedsld_baseline_2` | **Model:** `baseline` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSLD** achieved **88.89%** accuracy in Round 69 (runtime: 166.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDSLD** | `dirichlet_0.3` | 89.43% | **88.89%** | **89.43%** (R69) | 0.3050 | **88.89%** (R69) | R69 | 0.3050 | 166.2s |