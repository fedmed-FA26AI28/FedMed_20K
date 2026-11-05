# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-05 21:39:50  
**Run Name:** `pathmnist_dirichlet_0.1_fedsld_baseline_4` | **Model:** `baseline` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSLD** achieved **77.98%** accuracy in Round 66 (runtime: 604.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | 570,705 | 2229.3 KB | `sgd` | No | **FEDSLD** | `dirichlet_0.1` | 75.21% | **77.98%** | **73.80%** (R74) | 1.1232 | **77.98%** (R66) | R66 | 1.0817 | 604.2s |