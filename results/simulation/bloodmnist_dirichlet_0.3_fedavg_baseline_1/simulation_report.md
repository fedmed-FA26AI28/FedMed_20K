# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 10:11:14  
**Run Name:** `bloodmnist_dirichlet_0.3_fedavg_baseline_1` | **Model:** `baseline` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDAVG** achieved **91.03%** accuracy in Round 80 (runtime: 153.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDAVG** | `dirichlet_0.3` | 91.51% | **91.03%** | **91.51%** (R79) | 0.2343 | **91.03%** (R80) | R80 | 0.2343 | 153.3s |