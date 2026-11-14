# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 11:04:16  
**Run Name:** `bloodmnist_dirichlet_0.3_fedswitch_baseline_4` | **Model:** `baseline` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSWITCH** achieved **90.91%** accuracy in Round 75 (runtime: 158.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDSWITCH** | `dirichlet_0.3` | 90.70% | **90.91%** | **90.84%** (R75) | 0.2873 | **90.91%** (R75) | R75 | 0.2758 | 158.4s |