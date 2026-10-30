# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-30 10:18:03  
**Run Name:** `pathmnist_cla_baseline_bn_1` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] CLA** achieved **80.99%** accuracy in Round 75 (runtime: 1036.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **CLA** | `practical` | 79.82% | **80.99%** | **79.63%** (R77) | 1.0107 | **80.99%** (R75) | R75 | 1.0799 | 1036.2s |