# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-13 22:36:45  
**Run Name:** `bloodmnist_dirichlet_0.1_fedavg_baseline_3` | **Model:** `baseline` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDAVG** achieved **87.96%** accuracy in Round 75 (runtime: 175.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDAVG** | `dirichlet_0.1` | 88.12% | **87.96%** | **88.12%** (R75) | 0.3429 | **87.96%** (R75) | R75 | 0.3429 | 175.3s |