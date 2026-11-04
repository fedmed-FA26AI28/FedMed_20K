# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 20:00:42  
**Run Name:** `bloodmnist_dirichlet_0.3_fedbn_baseline_bn_4` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDBN** achieved **94.39%** accuracy in Round 77 (runtime: 97.1s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDBN** | `dirichlet_0.3` | 93.31% | **94.39%** | **93.85%** (R77) | 0.2024 | **94.39%** (R77) | R77 | 0.2571 | 97.1s |