# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 12:52:19  
**Run Name:** `pathmnist_dirichlet_0.1_run_2_seed_43_baseline_bn_fedavg` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDAVG** achieved **68.73%** accuracy in Round 68 (runtime: 698.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | `sgd` | No | **FEDAVG** | `dirichlet_0.1` | 52.26% | **68.73%** | **68.77%** (R68) | **68.73%** (R68) | R68 | 1.5926 | 698.4s |