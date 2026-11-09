# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-09 13:53:03  
**Run Name:** `pathmnist_practical_fedsol_baseline_bn_2` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] FEDSOL** achieved **76.59%** accuracy in Round 67 (runtime: 865.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDSOL** | `practical` | 76.02% | **76.59%** | **75.09%** (R74) | 1.3514 | **76.59%** (R67) | R67 | 1.3709 | 865.3s |