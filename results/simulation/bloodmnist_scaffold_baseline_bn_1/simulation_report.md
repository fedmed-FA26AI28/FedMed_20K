# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-30 10:37:01  
**Run Name:** `bloodmnist_scaffold_baseline_bn_1` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = practical:** **[BEST] SCAFFOLD** achieved **92.14%** accuracy in Round 73 (runtime: 164.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **SCAFFOLD** | `practical` | 91.90% | **92.14%** | **91.18%** (R54) | 0.2560 | **92.14%** (R73) | R73 | 0.2317 | 164.6s |