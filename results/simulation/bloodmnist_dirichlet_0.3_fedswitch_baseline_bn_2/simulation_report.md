# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 17:34:29  
**Run Name:** `bloodmnist_dirichlet_0.3_fedswitch_baseline_bn_2` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSWITCH** achieved **92.60%** accuracy in Round 79 (runtime: 166.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDSWITCH** | `dirichlet_0.3` | 87.81% | **92.60%** | **93.12%** (R79) | 0.2016 | **92.60%** (R79) | R79 | 0.3256 | 166.2s |