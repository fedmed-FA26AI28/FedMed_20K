# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-07 20:49:13  
**Run Name:** `pathmnist_practical_fedsol_baseline_2` | **Model:** `baseline` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] FEDSOL** achieved **79.05%** accuracy in Round 43 (runtime: 769.1s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | 570,705 | 2229.3 KB | `sgd` | No | **FEDSOL** | `practical` | 77.40% | **79.05%** | **77.64%** (R43) | 1.0541 | **79.05%** (R43) | R43 | 1.2443 | 769.1s |