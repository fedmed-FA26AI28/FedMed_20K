# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-09 14:07:39  
**Run Name:** `pathmnist_practical_fedsol_baseline_bn_3` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] FEDSOL** achieved **80.72%** accuracy in Round 76 (runtime: 861.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDSOL** | `practical` | 79.89% | **80.72%** | **76.10%** (R52) | 0.9086 | **80.72%** (R76) | R76 | 0.8246 | 861.2s |