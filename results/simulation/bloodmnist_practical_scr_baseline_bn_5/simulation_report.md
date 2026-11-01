# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-31 09:43:19  
**Run Name:** `scr_bn_bloodmnist_s46` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = practical:** **[BEST] SCR** achieved **87.34%** accuracy in Round 79 (runtime: 1339.1s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **SCR** | `practical` | 74.92% | **87.34%** | **85.54%** (R79) | 0.3798 | **87.34%** (R79) | R79 | 0.7421 | 1339.1s |