# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 10:09:55  
**Run Name:** `pathmnist_dirichlet_0.3_fedprox_baseline_1` | **Model:** `baseline` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDPROX** achieved **80.47%** accuracy in Round 30 (runtime: 490.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | 570,705 | 2229.3 KB | `sgd` | No | **FEDPROX** | `dirichlet_0.3` | 79.74% | **80.47%** | **79.90%** (R30) | 0.8449 | **80.47%** (R30) | R30 | 1.1558 | 490.6s |