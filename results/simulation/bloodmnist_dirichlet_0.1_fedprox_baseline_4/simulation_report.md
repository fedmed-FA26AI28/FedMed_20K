# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-13 22:56:27  
**Run Name:** `bloodmnist_dirichlet_0.1_fedprox_baseline_4` | **Model:** `baseline` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDPROX** achieved **86.32%** accuracy in Round 70 (runtime: 173.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDPROX** | `dirichlet_0.1` | 80.52% | **86.32%** | **80.52%** (R55) | 0.5627 | **86.32%** (R70) | R70 | 0.5627 | 173.4s |