# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-01 23:32:09  
**Run Name:** `pathmnist_practical_cla_baseline_bn_5` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] CLA** achieved **82.02%** accuracy in Round 79 (runtime: 695.1s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **CLA** | `practical` | 80.31% | **82.02%** | **80.55%** (R75) | 0.9495 | **82.02%** (R79) | R79 | 0.8948 | 695.1s |