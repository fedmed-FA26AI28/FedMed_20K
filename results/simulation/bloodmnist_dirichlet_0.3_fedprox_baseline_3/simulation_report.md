# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 10:32:01  
**Run Name:** `bloodmnist_dirichlet_0.3_fedprox_baseline_3` | **Model:** `baseline` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDPROX** achieved **89.97%** accuracy in Round 79 (runtime: 164.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDPROX** | `dirichlet_0.3` | 89.68% | **89.97%** | **89.68%** (R80) | 0.3027 | **89.97%** (R79) | R79 | 0.3027 | 164.5s |