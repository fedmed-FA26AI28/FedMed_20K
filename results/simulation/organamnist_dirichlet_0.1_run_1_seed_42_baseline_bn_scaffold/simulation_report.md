# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 09:14:54  
**Run Name:** `organamnist_dirichlet_0.1_run_1_seed_42_baseline_bn_scaffold` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] SCAFFOLD** achieved **84.77%** accuracy in Round 68 (runtime: 395.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | `sgd` | No | **SCAFFOLD** | `dirichlet_0.1` | 80.44% | **84.77%** | **80.59%** (R77) | **84.77%** (R68) | R68 | 0.6277 | 395.9s |