# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-31 09:20:34  
**Run Name:** `scr_bn_bloodmnist_s45` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = practical:** **[BEST] SCR** achieved **91.35%** accuracy in Round 73 (runtime: 1353.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **SCR** | `practical` | 90.73% | **91.35%** | **91.82%** (R62) | 0.2547 | **91.35%** (R73) | R73 | 0.2753 | 1353.9s |