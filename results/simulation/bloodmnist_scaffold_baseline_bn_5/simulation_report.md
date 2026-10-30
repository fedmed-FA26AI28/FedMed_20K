# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-30 10:48:04  
**Run Name:** `bloodmnist_scaffold_baseline_bn_5` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = practical:** **[BEST] SCAFFOLD** achieved **88.48%** accuracy in Round 79 (runtime: 137.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **SCAFFOLD** | `practical` | 87.14% | **88.48%** | **86.81%** (R79) | 0.3498 | **88.48%** (R79) | R79 | 0.3483 | 137.0s |