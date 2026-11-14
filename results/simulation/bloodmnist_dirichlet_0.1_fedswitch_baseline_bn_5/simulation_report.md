# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 15:39:14  
**Run Name:** `bloodmnist_dirichlet_0.1_fedswitch_baseline_bn_5` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSWITCH** achieved **89.59%** accuracy in Round 80 (runtime: 190.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDSWITCH** | `dirichlet_0.1` | 89.59% | **89.59%** | **86.36%** (R80) | 0.3650 | **89.59%** (R80) | R80 | 0.2898 | 190.7s |