# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-07 14:58:12  
**Run Name:** `pathmnist_dirichlet_0.1_fedsol_baseline_5` | **Model:** `baseline` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSOL** achieved **76.32%** accuracy in Round 51 (runtime: 838.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | 570,705 | 2229.3 KB | `sgd` | No | **FEDSOL** | `dirichlet_0.1` | 71.18% | **76.32%** | **77.83%** (R77) | 0.8275 | **76.32%** (R51) | R51 | 1.0439 | 838.5s |