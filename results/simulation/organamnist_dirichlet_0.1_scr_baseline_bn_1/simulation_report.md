# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-04 10:08:36  
**Run Name:** `dirichlet01_bn_organamnist_s42` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDAVG** achieved **84.89%** accuracy in Round 78 (runtime: 1901.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDAVG** | `dirichlet_0.1` | 80.44% | **84.89%** | **80.86%** (R78) | 0.6138 | **84.89%** (R78) | R78 | 0.6197 | 1901.6s |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **SCR** | `dirichlet_0.1` | 79.49% | **84.68%** | **80.93%** (R59) | 0.6365 | **84.68%** (R59) | R59 | 0.6234 | 1960.6s |