# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-01 22:12:49  
**Run Name:** `bloodmnist_practical_cla_baseline_bn_1` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = practical:** **[BEST] CLA** achieved **90.24%** accuracy in Round 68 (runtime: 152.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **CLA** | `practical` | 89.77% | **90.24%** | **88.36%** (R68) | 0.3402 | **90.24%** (R68) | R68 | 0.2948 | 152.6s |