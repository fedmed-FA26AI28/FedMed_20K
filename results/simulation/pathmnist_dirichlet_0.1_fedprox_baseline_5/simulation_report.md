# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-05 20:56:01  
**Run Name:** `pathmnist_dirichlet_0.1_fedprox_baseline_5` | **Model:** `baseline` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDPROX** achieved **77.65%** accuracy in Round 43 (runtime: 616.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | 570,705 | 2229.3 KB | `sgd` | No | **FEDPROX** | `dirichlet_0.1` | 71.94% | **77.65%** | **77.39%** (R29) | 1.2845 | **77.65%** (R43) | R43 | 1.2863 | 616.9s |