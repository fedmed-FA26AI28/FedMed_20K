# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-01 22:44:30  
**Run Name:** `pathmnist_practical_cla_baseline_bn_1` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] CLA** achieved **79.74%** accuracy in Round 75 (runtime: 1290.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **CLA** | `practical` | 78.44% | **79.74%** | **79.48%** (R35) | 1.1648 | **79.74%** (R75) | R75 | 1.2783 | 1290.4s |