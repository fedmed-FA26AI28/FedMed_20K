# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 19:57:06  
**Run Name:** `bloodmnist_dirichlet_0.3_fedbn_baseline_bn_2` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDBN** achieved **95.59%** accuracy in Round 68 (runtime: 103.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDBN** | `dirichlet_0.3` | 94.65% | **95.59%** | **95.68%** (R70) | 0.1442 | **95.59%** (R68) | R68 | 0.1722 | 103.0s |