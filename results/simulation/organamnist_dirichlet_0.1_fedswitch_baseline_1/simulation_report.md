# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-07 11:46:39  
**Run Name:** `organamnist_dirichlet_0.1_fedswitch_baseline_1` | **Model:** `baseline` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSWITCH** achieved **83.91%** accuracy in Round 79 (runtime: 313.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | 571,707 | 2233.2 KB | `sgd` | No | **FEDSWITCH** | `dirichlet_0.1` | 83.86% | **83.91%** | **81.39%** (R80) | 0.8081 | **83.91%** (R79) | R79 | 0.7092 | 313.9s |