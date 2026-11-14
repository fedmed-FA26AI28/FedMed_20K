# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 10:29:02  
**Run Name:** `bloodmnist_dirichlet_0.3_fedprox_baseline_2` | **Model:** `baseline` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDPROX** achieved **90.35%** accuracy in Round 77 (runtime: 171.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDPROX** | `dirichlet_0.3` | 90.83% | **90.35%** | **90.83%** (R77) | 0.2735 | **90.35%** (R77) | R77 | 0.2735 | 171.5s |