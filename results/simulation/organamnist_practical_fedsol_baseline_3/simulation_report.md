# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-07 16:29:33  
**Run Name:** `organamnist_practical_fedsol_baseline_3` | **Model:** `baseline` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = practical:** **[BEST] FEDSOL** achieved **83.46%** accuracy in Round 69 (runtime: 356.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | 571,707 | 2233.2 KB | `sgd` | No | **FEDSOL** | `practical` | 83.20% | **83.46%** | **84.19%** (R69) | 0.6664 | **83.46%** (R69) | R69 | 0.7238 | 356.4s |