# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 12:14:51  
**Run Name:** `bloodmnist_dirichlet_0.1_run_5_seed_46_baseline_bn_fedavg` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDAVG** achieved **89.77%** accuracy in Round 80 (runtime: 134.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | `sgd` | No | **FEDAVG** | `dirichlet_0.1` | 89.77% | **89.77%** | **87.63%** (R78) | **89.77%** (R80) | R80 | 0.2772 | 134.3s |