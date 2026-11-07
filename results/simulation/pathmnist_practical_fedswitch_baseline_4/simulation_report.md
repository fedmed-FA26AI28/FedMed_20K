# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-07 20:08:13  
**Run Name:** `pathmnist_practical_fedswitch_baseline_4` | **Model:** `baseline` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] FEDSWITCH** achieved **80.28%** accuracy in Round 28 (runtime: 706.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | 570,705 | 2229.3 KB | `sgd` | No | **FEDSWITCH** | `practical` | 78.23% | **80.28%** | **79.21%** (R28) | 0.9497 | **80.28%** (R28) | R28 | 1.1149 | 706.6s |