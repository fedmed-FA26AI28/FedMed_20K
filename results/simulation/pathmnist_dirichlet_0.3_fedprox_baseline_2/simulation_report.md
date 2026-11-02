# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 10:19:23  
**Run Name:** `pathmnist_dirichlet_0.3_fedprox_baseline_2` | **Model:** `baseline` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDPROX** achieved **78.48%** accuracy in Round 68 (runtime: 553.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | 570,705 | 2229.3 KB | `sgd` | No | **FEDPROX** | `dirichlet_0.3` | 77.65% | **78.48%** | **76.50%** (R75) | 1.7553 | **78.48%** (R68) | R68 | 1.7173 | 553.4s |