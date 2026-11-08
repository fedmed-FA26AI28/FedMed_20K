# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-08 13:45:47  
**Run Name:** `pathmnist_dirichlet_0.3_fedswitch_baseline_3` | **Model:** `baseline` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSWITCH** achieved **79.11%** accuracy in Round 67 (runtime: 476.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | 570,705 | 2229.3 KB | `sgd` | No | **FEDSWITCH** | `dirichlet_0.3` | 77.62% | **79.11%** | **78.32%** (R67) | 1.2016 | **79.11%** (R67) | R67 | 1.3887 | 476.9s |