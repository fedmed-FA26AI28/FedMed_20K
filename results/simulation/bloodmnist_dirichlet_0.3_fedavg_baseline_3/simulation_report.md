# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 10:17:08  
**Run Name:** `bloodmnist_dirichlet_0.3_fedavg_baseline_3` | **Model:** `baseline` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDAVG** achieved **89.83%** accuracy in Round 79 (runtime: 158.1s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDAVG** | `dirichlet_0.3` | 89.68% | **89.83%** | **89.68%** (R80) | 0.3014 | **89.83%** (R79) | R79 | 0.3014 | 158.1s |