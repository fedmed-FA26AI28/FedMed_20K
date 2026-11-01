# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-31 18:57:23  
**Run Name:** `scr_bn_organamnist_s45` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = practical:** **[BEST] SCR** achieved **85.94%** accuracy in Round 78 (runtime: 4649.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **SCR** | `practical` | 85.50% | **85.94%** | **86.84%** (R76) | 0.4789 | **85.94%** (R78) | R78 | 0.5192 | 4649.7s |