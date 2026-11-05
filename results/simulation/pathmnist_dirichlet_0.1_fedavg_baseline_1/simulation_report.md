# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-05 19:17:55  
**Run Name:** `pathmnist_dirichlet_0.1_fedavg_baseline_1` | **Model:** `baseline` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDAVG** achieved **75.74%** accuracy in Round 78 (runtime: 632.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | 570,705 | 2229.3 KB | `sgd` | No | **FEDAVG** | `dirichlet_0.1` | 75.04% | **75.74%** | **76.93%** (R78) | 1.2605 | **75.74%** (R78) | R78 | 1.3297 | 632.5s |