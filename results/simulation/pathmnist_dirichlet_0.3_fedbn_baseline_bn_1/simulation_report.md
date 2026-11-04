# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 20:18:32  
**Run Name:** `pathmnist_dirichlet_0.3_fedbn_baseline_bn_1` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDBN** achieved **84.65%** accuracy in Round 78 (runtime: 944.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDBN** | `dirichlet_0.3` | 83.72% | **84.65%** | **82.94%** (R72) | 0.9774 | **84.65%** (R78) | R78 | 0.9575 | 944.8s |