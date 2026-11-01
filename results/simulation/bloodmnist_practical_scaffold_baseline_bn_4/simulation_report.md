# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-30 10:45:32  
**Run Name:** `bloodmnist_scaffold_baseline_bn_4` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = practical:** **[BEST] SCAFFOLD** achieved **89.71%** accuracy in Round 73 (runtime: 135.1s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **SCAFFOLD** | `practical` | 88.86% | **89.71%** | **89.54%** (R73) | 0.2934 | **89.71%** (R73) | R73 | 0.3041 | 135.1s |