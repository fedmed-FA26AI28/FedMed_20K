# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-07 13:32:18  
**Run Name:** `pathmnist_dirichlet_0.1_fedswitch_baseline_4` | **Model:** `baseline` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSWITCH** achieved **76.07%** accuracy in Round 58 (runtime: 650.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | 570,705 | 2229.3 KB | `sgd` | No | **FEDSWITCH** | `dirichlet_0.1` | 72.34% | **76.07%** | **73.35%** (R30) | 0.8693 | **76.07%** (R58) | R58 | 1.3968 | 650.7s |