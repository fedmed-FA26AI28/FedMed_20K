# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 11:23:23  
**Run Name:** `bloodmnist_dirichlet_0.3_fedsol_baseline_5` | **Model:** `baseline` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSOL** achieved **92.14%** accuracy in Round 80 (runtime: 180.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDSOL** | `dirichlet_0.3` | 92.14% | **92.14%** | **92.49%** (R74) | 0.2422 | **92.14%** (R80) | R80 | 0.2542 | 180.8s |