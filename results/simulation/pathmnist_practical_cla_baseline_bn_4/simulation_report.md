# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-01 23:20:18  
**Run Name:** `pathmnist_practical_cla_baseline_bn_4` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] CLA** achieved **83.04%** accuracy in Round 80 (runtime: 797.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **CLA** | `practical` | 83.04% | **83.04%** | **81.34%** (R79) | 1.0138 | **83.04%** (R80) | R80 | 0.8720 | 797.8s |