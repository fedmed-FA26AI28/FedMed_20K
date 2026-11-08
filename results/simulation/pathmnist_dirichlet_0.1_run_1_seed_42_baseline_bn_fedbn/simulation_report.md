# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 13:39:11  
**Run Name:** `pathmnist_dirichlet_0.1_run_1_seed_42_baseline_bn_fedbn` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDBN** achieved **82.92%** accuracy in Round 75 (runtime: 658.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | `sgd` | No | **FEDBN** | `dirichlet_0.1` | 82.92% | **82.92%** | **83.38%** (R65) | **82.92%** (R75) | R75 | 0.9678 | 658.5s |