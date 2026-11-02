# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 09:08:05  
**Run Name:** `organamnist_dirichlet_0.1_run_1_seed_42_baseline_bn_fedbn` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDBN** achieved **93.51%** accuracy in Round 78 (runtime: 270.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | `sgd` | No | **FEDBN** | `dirichlet_0.1` | 88.41% | **93.51%** | **92.51%** (R78) | **93.51%** (R78) | R78 | 0.4189 | 270.0s |