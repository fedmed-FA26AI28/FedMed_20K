# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 20:28:24  
**Run Name:** `pathmnist_dirichlet_0.3_fedbn_baseline_bn_2` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDBN** achieved **85.39%** accuracy in Round 72 (runtime: 577.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDBN** | `dirichlet_0.3` | 82.72% | **85.39%** | **83.64%** (R72) | 0.7420 | **85.39%** (R72) | R72 | 0.8906 | 577.0s |