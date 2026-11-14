# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 17:53:34  
**Run Name:** `bloodmnist_dirichlet_0.3_fedsol_baseline_bn_3` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSOL** achieved **88.45%** accuracy in Round 68 (runtime: 190.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDSOL** | `dirichlet_0.3` | 87.99% | **88.45%** | **88.62%** (R68) | 0.3608 | **88.45%** (R68) | R68 | 0.3619 | 190.3s |