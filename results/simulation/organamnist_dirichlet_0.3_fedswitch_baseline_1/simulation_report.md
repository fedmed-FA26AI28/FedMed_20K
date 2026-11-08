# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-08 09:49:54  
**Run Name:** `organamnist_dirichlet_0.3_fedswitch_baseline_1` | **Model:** `baseline` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSWITCH** achieved **83.49%** accuracy in Round 58 (runtime: 257.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | 571,707 | 2233.2 KB | `sgd` | No | **FEDSWITCH** | `dirichlet_0.3` | 83.33% | **83.49%** | **82.65%** (R77) | 0.9138 | **83.49%** (R58) | R58 | 0.8715 | 257.6s |