# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 14:19:01  
**Run Name:** `pathmnist_dirichlet_0.3_fedsld_baseline_bn_1` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSLD** achieved **83.69%** accuracy in Round 66 (runtime: 495.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDSLD** | `dirichlet_0.3` | 79.16% | **83.69%** | **83.31%** (R66) | 0.6553 | **83.69%** (R66) | R66 | 0.7902 | 495.7s |