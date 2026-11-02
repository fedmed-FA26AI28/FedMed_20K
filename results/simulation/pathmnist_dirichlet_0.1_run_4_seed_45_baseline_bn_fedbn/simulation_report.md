# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 14:12:03  
**Run Name:** `pathmnist_dirichlet_0.1_run_4_seed_45_baseline_bn_fedbn` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDBN** achieved **80.68%** accuracy in Round 71 (runtime: 603.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | `sgd` | No | **FEDBN** | `dirichlet_0.1` | 54.18% | **80.68%** | **78.22%** (R71) | **80.68%** (R71) | R71 | 2.7507 | 603.4s |