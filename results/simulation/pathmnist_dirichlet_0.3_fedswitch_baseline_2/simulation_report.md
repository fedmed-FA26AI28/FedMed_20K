# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-08 13:37:36  
**Run Name:** `pathmnist_dirichlet_0.3_fedswitch_baseline_2` | **Model:** `baseline` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSWITCH** achieved **79.30%** accuracy in Round 71 (runtime: 543.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | 570,705 | 2229.3 KB | `sgd` | No | **FEDSWITCH** | `dirichlet_0.3` | 78.29% | **79.30%** | **77.81%** (R67) | 1.5842 | **79.30%** (R71) | R71 | 1.6696 | 543.9s |