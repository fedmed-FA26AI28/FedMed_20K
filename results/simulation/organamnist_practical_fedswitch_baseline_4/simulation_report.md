# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-07 18:44:58  
**Run Name:** `organamnist_practical_fedswitch_baseline_4` | **Model:** `baseline` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = practical:** **[BEST] FEDSWITCH** achieved **83.68%** accuracy in Round 75 (runtime: 314.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | 571,707 | 2233.2 KB | `sgd` | No | **FEDSWITCH** | `practical` | 83.61% | **83.68%** | **83.73%** (R75) | 0.7576 | **83.68%** (R75) | R75 | 0.7701 | 314.5s |