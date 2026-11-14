# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 13:23:38  
**Run Name:** `bloodmnist_practical_fedswitch_baseline_bn_4` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = practical:** **[BEST] FEDSWITCH** achieved **89.13%** accuracy in Round 79 (runtime: 160.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDSWITCH** | `practical` | 88.72% | **89.13%** | **88.89%** (R79) | 0.3025 | **89.13%** (R79) | R79 | 0.3020 | 160.7s |