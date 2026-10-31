# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-31 08:57:34  
**Run Name:** `scr_bn_bloodmnist_s44` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = practical:** **[BEST] SCR** achieved **91.52%** accuracy in Round 71 (runtime: 1398.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **SCR** | `practical` | 91.03% | **91.52%** | **92.06%** (R74) | 0.2534 | **91.52%** (R71) | R71 | 0.2802 | 1398.0s |