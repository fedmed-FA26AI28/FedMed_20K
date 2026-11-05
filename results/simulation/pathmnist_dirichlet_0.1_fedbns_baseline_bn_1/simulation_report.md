# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-05 17:16:25  
**Run Name:** `pathmnist_dirichlet_0.1_fedbns_baseline_bn_1` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDBNS** achieved **86.06%** accuracy in Round 75 (runtime: 690.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDBNS** | `dirichlet_0.1` | 85.82% | **86.06%** | **86.61%** (R75) | 0.8735 | **86.06%** (R75) | R75 | 1.0766 | 690.0s |