# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-07 17:26:30  
**Run Name:** `pathmnist_dirichlet_0.3_fedsld_baseline_1` | **Model:** `baseline` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSLD** achieved **78.69%** accuracy in Round 20 (runtime: 552.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | 570,705 | 2229.3 KB | `sgd` | No | **FEDSLD** | `dirichlet_0.3` | 77.07% | **78.69%** | **77.07%** (R73) | 0.8191 | **78.69%** (R20) | R20 | 0.8191 | 552.2s |