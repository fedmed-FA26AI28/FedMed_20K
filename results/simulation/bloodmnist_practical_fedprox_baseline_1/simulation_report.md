# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-13 21:23:04  
**Run Name:** `bloodmnist_practical_fedprox_baseline_1` | **Model:** `baseline` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = practical:** **[BEST] FEDPROX** achieved **88.28%** accuracy in Round 77 (runtime: 196.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDPROX** | `practical` | 85.14% | **88.28%** | **85.14%** (R78) | 0.4073 | **88.28%** (R77) | R77 | 0.4073 | 196.9s |