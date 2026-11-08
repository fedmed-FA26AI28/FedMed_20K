# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-08 09:57:59  
**Run Name:** `organamnist_dirichlet_0.3_fedswitch_baseline_3` | **Model:** `baseline` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSWITCH** achieved **84.81%** accuracy in Round 67 (runtime: 246.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | 571,707 | 2233.2 KB | `sgd` | No | **FEDSWITCH** | `dirichlet_0.3` | 84.66% | **84.81%** | **85.04%** (R80) | 0.7891 | **84.81%** (R67) | R67 | 0.8290 | 246.3s |