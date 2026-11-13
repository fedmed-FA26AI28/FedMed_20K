# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-13 23:26:02  
**Run Name:** `bloodmnist_dirichlet_0.1_fedswitch_baseline_3` | **Model:** `baseline` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSWITCH** achieved **87.99%** accuracy in Round 78 (runtime: 171.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDSWITCH** | `dirichlet_0.1` | 87.40% | **87.99%** | **88.19%** (R78) | 0.3317 | **87.99%** (R78) | R78 | 0.3536 | 171.3s |