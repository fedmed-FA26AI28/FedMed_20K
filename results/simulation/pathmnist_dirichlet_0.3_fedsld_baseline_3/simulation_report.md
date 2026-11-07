# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-07 17:46:40  
**Run Name:** `pathmnist_dirichlet_0.3_fedsld_baseline_3` | **Model:** `baseline` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSLD** achieved **79.28%** accuracy in Round 48 (runtime: 556.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | 570,705 | 2229.3 KB | `sgd` | No | **FEDSLD** | `dirichlet_0.3` | 79.08% | **79.28%** | **79.08%** (R48) | 0.8303 | **79.28%** (R48) | R48 | 0.8303 | 556.8s |