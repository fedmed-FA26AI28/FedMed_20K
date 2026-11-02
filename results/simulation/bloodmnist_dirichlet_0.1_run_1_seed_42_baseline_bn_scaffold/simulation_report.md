# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 11:36:51  
**Run Name:** `bloodmnist_dirichlet_0.1_run_1_seed_42_baseline_bn_scaffold` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] SCAFFOLD** achieved **91.93%** accuracy in Round 79 (runtime: 131.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | `sgd` | No | **SCAFFOLD** | `dirichlet_0.1` | 90.50% | **91.93%** | **92.27%** (R79) | **91.93%** (R79) | R79 | 0.2655 | 131.3s |