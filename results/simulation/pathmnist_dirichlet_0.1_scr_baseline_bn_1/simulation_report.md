# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-04 17:00:03  
**Run Name:** `dirichlet01_bn_pathmnist_s42` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] SCR** achieved **76.41%** accuracy in Round 78 (runtime: 4057.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **SCR** | `dirichlet_0.1` | 74.68% | **76.41%** | **76.26%** (R78) | 1.0791 | **76.41%** (R78) | R78 | 1.1893 | 4057.2s |