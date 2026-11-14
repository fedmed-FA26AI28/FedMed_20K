# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 10:14:14  
**Run Name:** `bloodmnist_dirichlet_0.3_fedavg_baseline_2` | **Model:** `baseline` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDAVG** achieved **90.47%** accuracy in Round 73 (runtime: 165.1s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDAVG** | `dirichlet_0.3` | 90.79% | **90.47%** | **90.79%** (R73) | 0.2828 | **90.47%** (R73) | R73 | 0.2828 | 165.1s |