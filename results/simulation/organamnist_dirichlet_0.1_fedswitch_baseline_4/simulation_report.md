# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-07 12:02:16  
**Run Name:** `organamnist_dirichlet_0.1_fedswitch_baseline_4` | **Model:** `baseline` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSWITCH** achieved **82.47%** accuracy in Round 76 (runtime: 277.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | 571,707 | 2233.2 KB | `sgd` | No | **FEDSWITCH** | `dirichlet_0.1` | 82.35% | **82.47%** | **80.62%** (R79) | 0.9094 | **82.47%** (R76) | R76 | 0.8013 | 277.4s |