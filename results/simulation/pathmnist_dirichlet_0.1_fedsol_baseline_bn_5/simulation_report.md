# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-09 16:58:19  
**Run Name:** `pathmnist_dirichlet_0.1_fedsol_baseline_bn_5` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSOL** achieved **76.21%** accuracy in Round 70 (runtime: 918.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDSOL** | `dirichlet_0.1` | 72.48% | **76.21%** | **77.98%** (R58) | 0.8474 | **76.21%** (R70) | R70 | 1.0213 | 918.7s |