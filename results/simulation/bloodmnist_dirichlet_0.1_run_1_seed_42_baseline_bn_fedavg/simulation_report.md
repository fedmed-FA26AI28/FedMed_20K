# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 11:30:23  
**Run Name:** `bloodmnist_dirichlet_0.1_run_1_seed_42_baseline_bn_fedavg` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDAVG** achieved **91.32%** accuracy in Round 78 (runtime: 115.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | `sgd` | No | **FEDAVG** | `dirichlet_0.1` | 90.50% | **91.32%** | **92.42%** (R78) | **91.32%** (R78) | R78 | 0.2770 | 115.6s |