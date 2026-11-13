# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-13 21:39:58  
**Run Name:** `bloodmnist_practical_fedsld_baseline_1` | **Model:** `baseline` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = practical:** **[BEST] FEDSLD** achieved **83.57%** accuracy in Round 76 (runtime: 201.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDSLD** | `practical` | 84.64% | **83.57%** | **84.64%** (R76) | 0.4475 | **83.57%** (R76) | R76 | 0.4475 | 201.7s |