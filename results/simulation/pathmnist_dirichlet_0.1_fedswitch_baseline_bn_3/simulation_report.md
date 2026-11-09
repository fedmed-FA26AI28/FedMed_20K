# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-09 15:15:15  
**Run Name:** `pathmnist_dirichlet_0.1_fedswitch_baseline_bn_3` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSWITCH** achieved **75.36%** accuracy in Round 66 (runtime: 674.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDSWITCH** | `dirichlet_0.1` | 74.80% | **75.36%** | **79.41%** (R48) | 0.8060 | **75.36%** (R66) | R66 | 1.1576 | 674.8s |