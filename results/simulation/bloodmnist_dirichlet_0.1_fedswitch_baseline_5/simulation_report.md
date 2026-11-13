# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-13 23:32:18  
**Run Name:** `bloodmnist_dirichlet_0.1_fedswitch_baseline_5` | **Model:** `baseline` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSWITCH** achieved **88.31%** accuracy in Round 80 (runtime: 181.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDSWITCH** | `dirichlet_0.1` | 88.31% | **88.31%** | **86.65%** (R80) | 0.3918 | **88.31%** (R80) | R80 | 0.3293 | 181.5s |