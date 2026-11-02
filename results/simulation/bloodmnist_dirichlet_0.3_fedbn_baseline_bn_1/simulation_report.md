# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 19:55:08  
**Run Name:** `bloodmnist_dirichlet_0.3_fedbn_baseline_bn_1` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDBN** achieved **95.03%** accuracy in Round 78 (runtime: 86.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDBN** | `dirichlet_0.3` | 94.48% | **95.03%** | **95.16%** (R78) | 0.1545 | **95.03%** (R78) | R78 | 0.1942 | 86.5s |