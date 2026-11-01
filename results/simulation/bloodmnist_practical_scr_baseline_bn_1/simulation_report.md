# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-31 08:10:43  
**Run Name:** `scr_bn_bloodmnist_s42` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = practical:** **[BEST] SCR** achieved **91.17%** accuracy in Round 71 (runtime: 1427.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **SCR** | `practical` | 89.45% | **91.17%** | **90.50%** (R71) | 0.2574 | **91.17%** (R71) | R71 | 0.3143 | 1427.4s |