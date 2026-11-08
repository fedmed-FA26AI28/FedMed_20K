# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 13:04:04  
**Run Name:** `pathmnist_dirichlet_0.1_run_3_seed_44_baseline_bn_fedavg` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDAVG** achieved **78.83%** accuracy in Round 37 (runtime: 691.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | `sgd` | No | **FEDAVG** | `dirichlet_0.1` | 71.60% | **78.83%** | **77.53%** (R37) | **78.83%** (R37) | R37 | 1.0498 | 691.0s |