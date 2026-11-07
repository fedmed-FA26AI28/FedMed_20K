# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-07 18:34:22  
**Run Name:** `organamnist_practical_fedswitch_baseline_2` | **Model:** `baseline` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = practical:** **[BEST] FEDSWITCH** achieved **83.04%** accuracy in Round 80 (runtime: 320.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | 571,707 | 2233.2 KB | `sgd` | No | **FEDSWITCH** | `practical` | 83.04% | **83.04%** | **82.40%** (R77) | 0.7985 | **83.04%** (R80) | R80 | 0.7780 | 320.3s |