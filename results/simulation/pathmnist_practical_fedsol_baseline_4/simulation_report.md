# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-07 21:17:19  
**Run Name:** `pathmnist_practical_fedsol_baseline_4` | **Model:** `baseline` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] FEDSOL** achieved **80.79%** accuracy in Round 53 (runtime: 891.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | 570,705 | 2229.3 KB | `sgd` | No | **FEDSOL** | `practical` | 80.20% | **80.79%** | **80.09%** (R71) | 0.8888 | **80.79%** (R53) | R53 | 0.9307 | 891.8s |