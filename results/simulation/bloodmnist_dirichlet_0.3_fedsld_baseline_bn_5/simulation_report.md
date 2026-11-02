# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 13:47:51  
**Run Name:** `bloodmnist_dirichlet_0.3_fedsld_baseline_bn_5` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSLD** achieved **93.13%** accuracy in Round 67 (runtime: 99.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDSLD** | `dirichlet_0.3` | 85.85% | **93.13%** | **93.30%** (R67) | 0.1967 | **93.13%** (R67) | R67 | 0.3861 | 99.3s |