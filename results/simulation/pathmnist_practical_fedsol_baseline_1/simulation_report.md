# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-07 20:36:08  
**Run Name:** `pathmnist_practical_fedsol_baseline_1` | **Model:** `baseline` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] FEDSOL** achieved **79.00%** accuracy in Round 31 (runtime: 976.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | 570,705 | 2229.3 KB | `sgd` | No | **FEDSOL** | `practical` | 77.99% | **79.00%** | **79.66%** (R59) | 0.9792 | **79.00%** (R31) | R31 | 1.1148 | 976.7s |