# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-09 17:06:56  
**Run Name:** `pathmnist_dirichlet_0.3_fedswitch_baseline_bn_1` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSWITCH** achieved **84.04%** accuracy in Round 62 (runtime: 502.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDSWITCH** | `dirichlet_0.3` | 82.74% | **84.04%** | **83.98%** (R62) | 0.8662 | **84.04%** (R62) | R62 | 0.8984 | 502.8s |