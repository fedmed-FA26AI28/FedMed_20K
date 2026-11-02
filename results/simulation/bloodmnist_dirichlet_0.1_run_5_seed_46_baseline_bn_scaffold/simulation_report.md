# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 12:22:09  
**Run Name:** `bloodmnist_dirichlet_0.1_run_5_seed_46_baseline_bn_scaffold` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] SCAFFOLD** achieved **90.91%** accuracy in Round 79 (runtime: 143.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | `sgd` | No | **SCAFFOLD** | `dirichlet_0.1` | 90.41% | **90.91%** | **88.86%** (R79) | **90.91%** (R79) | R79 | 0.2671 | 143.9s |