# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 15:32:49  
**Run Name:** `bloodmnist_dirichlet_0.1_fedswitch_baseline_bn_3` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSWITCH** achieved **88.22%** accuracy in Round 78 (runtime: 170.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDSWITCH** | `dirichlet_0.1` | 86.44% | **88.22%** | **89.78%** (R72) | 0.3093 | **88.22%** (R78) | R78 | 0.3748 | 170.3s |