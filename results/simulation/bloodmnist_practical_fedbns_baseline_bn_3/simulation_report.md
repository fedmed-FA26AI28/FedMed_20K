# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 12:48:03  
**Run Name:** `bloodmnist_practical_fedbns_baseline_bn_3` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = practical:** **[BEST] FEDBNS** achieved **96.70%** accuracy in Round 80 (runtime: 200.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDBNS** | `practical` | 96.70% | **96.70%** | **94.65%** (R80) | 0.1819 | **96.70%** (R80) | R80 | 0.1235 | 200.0s |