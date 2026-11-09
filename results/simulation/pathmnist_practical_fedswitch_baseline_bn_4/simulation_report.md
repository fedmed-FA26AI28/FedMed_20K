# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-09 13:09:26  
**Run Name:** `pathmnist_practical_fedswitch_baseline_bn_4` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] FEDSWITCH** achieved **84.32%** accuracy in Round 62 (runtime: 681.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDSWITCH** | `practical` | 83.86% | **84.32%** | **84.31%** (R62) | 0.9653 | **84.32%** (R62) | R62 | 0.8726 | 681.5s |