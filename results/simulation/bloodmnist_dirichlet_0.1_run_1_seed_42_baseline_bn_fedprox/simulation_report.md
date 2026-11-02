# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 11:32:37  
**Run Name:** `bloodmnist_dirichlet_0.1_run_1_seed_42_baseline_bn_fedprox` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDPROX** achieved **91.32%** accuracy in Round 67 (runtime: 121.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | `sgd` | No | **FEDPROX** | `dirichlet_0.1` | 90.56% | **91.32%** | **91.88%** (R71) | **91.32%** (R67) | R67 | 0.2684 | 121.6s |