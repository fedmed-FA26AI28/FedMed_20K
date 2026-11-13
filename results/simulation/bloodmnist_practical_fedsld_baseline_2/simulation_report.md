# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-13 21:43:21  
**Run Name:** `bloodmnist_practical_fedsld_baseline_2` | **Model:** `baseline` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = practical:** **[BEST] FEDSLD** achieved **76.29%** accuracy in Round 80 (runtime: 187.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDSLD** | `practical` | 71.23% | **76.29%** | **71.23%** (R74) | 0.9121 | **76.29%** (R80) | R80 | 0.9121 | 187.3s |