# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-07 16:23:20  
**Run Name:** `organamnist_practical_fedsol_baseline_2` | **Model:** `baseline` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = practical:** **[BEST] FEDSOL** achieved **83.56%** accuracy in Round 80 (runtime: 392.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | 571,707 | 2233.2 KB | `sgd` | No | **FEDSOL** | `practical` | 83.56% | **83.56%** | **82.58%** (R80) | 0.7948 | **83.56%** (R80) | R80 | 0.7504 | 392.7s |