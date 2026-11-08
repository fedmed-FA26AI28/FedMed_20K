# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-08 14:38:05  
**Run Name:** `pathmnist_dirichlet_0.3_fedsol_baseline_3` | **Model:** `baseline` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSOL** achieved **80.38%** accuracy in Round 60 (runtime: 709.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | 570,705 | 2229.3 KB | `sgd` | No | **FEDSOL** | `dirichlet_0.3` | 79.97% | **80.38%** | **80.20%** (R20) | 0.9797 | **80.38%** (R60) | R60 | 1.3856 | 709.4s |