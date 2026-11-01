# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-31 16:21:33  
**Run Name:** `scr_bn_organamnist_s43` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = practical:** **[BEST] SCR** achieved **86.39%** accuracy in Round 72 (runtime: 4642.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **SCR** | `practical` | 86.32% | **86.39%** | **85.77%** (R66) | 0.5588 | **86.39%** (R72) | R72 | 0.5528 | 4642.9s |