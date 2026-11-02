# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 11:41:25  
**Run Name:** `bloodmnist_dirichlet_0.1_run_2_seed_43_baseline_bn_fedavg` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDAVG** achieved **86.73%** accuracy in Round 77 (runtime: 139.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | `sgd` | No | **FEDAVG** | `dirichlet_0.1` | 80.06% | **86.73%** | **88.13%** (R73) | **86.73%** (R77) | R77 | 0.5151 | 139.3s |