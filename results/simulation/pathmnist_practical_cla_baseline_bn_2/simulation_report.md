# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-01 22:55:46  
**Run Name:** `pathmnist_practical_cla_baseline_bn_2` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] CLA** achieved **84.39%** accuracy in Round 80 (runtime: 660.1s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **CLA** | `practical` | 84.39% | **84.39%** | **82.62%** (R80) | 0.7342 | **84.39%** (R80) | R80 | 0.6400 | 660.1s |