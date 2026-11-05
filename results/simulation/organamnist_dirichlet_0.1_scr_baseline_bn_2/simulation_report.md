# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-04 11:50:54  
**Run Name:** `dirichlet01_bn_organamnist_s43` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] SCR** achieved **84.66%** accuracy in Round 77 (runtime: 1960.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDAVG** | `dirichlet_0.1` | 84.26% | **84.52%** | **84.44%** (R68) | 0.4998 | **84.52%** (R72) | R72 | 0.5356 | 2005.0s |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **SCR** | `dirichlet_0.1` | 84.35% | **84.66%** | **83.83%** (R77) | 0.5306 | **84.66%** (R77) | R77 | 0.5474 | 1960.0s |