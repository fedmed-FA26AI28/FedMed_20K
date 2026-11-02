# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 10:27:56  
**Run Name:** `pathmnist_dirichlet_0.3_fedprox_baseline_3` | **Model:** `baseline` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDPROX** achieved **79.81%** accuracy in Round 79 (runtime: 499.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | 570,705 | 2229.3 KB | `sgd` | No | **FEDPROX** | `dirichlet_0.3` | 79.57% | **79.81%** | **78.82%** (R80) | 1.3872 | **79.81%** (R79) | R79 | 1.3364 | 499.2s |