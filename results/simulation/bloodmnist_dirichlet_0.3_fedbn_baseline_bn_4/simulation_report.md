# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 16:39:48  
**Run Name:** `bloodmnist_dirichlet_0.3_fedbn_baseline_bn_4` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDBN** achieved **94.10%** accuracy in Round 80 (runtime: 160.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDBN** | `dirichlet_0.3` | 94.10% | **94.10%** | **93.80%** (R78) | 0.2110 | **94.10%** (R80) | R80 | 0.2062 | 160.3s |