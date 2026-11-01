# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-31 17:39:22  
**Run Name:** `scr_bn_organamnist_s44` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = practical:** **[BEST] SCR** achieved **85.95%** accuracy in Round 75 (runtime: 4638.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **SCR** | `practical` | 85.51% | **85.95%** | **86.36%** (R75) | 0.4983 | **85.95%** (R75) | R75 | 0.5155 | 4638.5s |