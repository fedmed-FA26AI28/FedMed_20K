# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-09 17:16:50  
**Run Name:** `pathmnist_dirichlet_0.3_fedswitch_baseline_bn_2` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSWITCH** achieved **84.22%** accuracy in Round 51 (runtime: 580.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDSWITCH** | `dirichlet_0.3` | 83.59% | **84.22%** | **82.90%** (R51) | 0.8638 | **84.22%** (R51) | R51 | 0.8939 | 580.2s |