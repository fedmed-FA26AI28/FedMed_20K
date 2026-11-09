# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-09 12:48:10  
**Run Name:** `pathmnist_practical_fedswitch_baseline_bn_2` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] FEDSWITCH** achieved **84.61%** accuracy in Round 66 (runtime: 556.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDSWITCH** | `practical` | 84.55% | **84.61%** | **83.61%** (R80) | 0.8489 | **84.61%** (R66) | R66 | 0.7739 | 556.6s |