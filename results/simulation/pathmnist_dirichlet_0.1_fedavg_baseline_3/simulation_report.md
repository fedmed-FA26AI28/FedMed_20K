# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-05 19:39:43  
**Run Name:** `pathmnist_dirichlet_0.1_fedavg_baseline_3` | **Model:** `baseline` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDAVG** achieved **78.40%** accuracy in Round 75 (runtime: 637.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | 570,705 | 2229.3 KB | `sgd` | No | **FEDAVG** | `dirichlet_0.1` | 71.78% | **78.40%** | **76.55%** (R79) | 1.2402 | **78.40%** (R75) | R75 | 1.1587 | 637.5s |