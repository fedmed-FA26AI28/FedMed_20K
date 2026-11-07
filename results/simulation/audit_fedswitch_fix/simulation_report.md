# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-07 17:30:02  
**Run Name:** `audit_fedswitch_fix` | **Model:** `baseline` | **Seed:** `42` | **Rounds:** 3 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSWITCH** achieved **49.14%** accuracy in Round 3 (runtime: 26.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDSWITCH** | `dirichlet_0.3` | 49.14% | **49.14%** | **49.17%** (R3) | 1.2150 | **49.14%** (R3) | R3 | 1.2247 | 26.5s |