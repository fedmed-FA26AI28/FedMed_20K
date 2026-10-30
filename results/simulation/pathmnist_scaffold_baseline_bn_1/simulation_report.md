# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-30 11:32:49  
**Run Name:** `pathmnist_scaffold_baseline_bn_1` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] SCAFFOLD** achieved **80.91%** accuracy in Round 65 (runtime: 850.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **SCAFFOLD** | `practical` | 78.30% | **80.91%** | **80.12%** (R65) | 0.8672 | **80.91%** (R65) | R65 | 0.9984 | 850.4s |