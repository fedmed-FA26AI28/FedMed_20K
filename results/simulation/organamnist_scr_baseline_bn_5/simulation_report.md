# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-31 20:15:00  
**Run Name:** `scr_bn_organamnist_s46` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = practical:** **[BEST] SCR** achieved **85.17%** accuracy in Round 79 (runtime: 4626.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **SCR** | `practical` | 85.13% | **85.17%** | **85.22%** (R80) | 0.5949 | **85.17%** (R79) | R79 | 0.6036 | 4626.5s |