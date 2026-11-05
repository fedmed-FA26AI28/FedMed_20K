# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-05 16:43:12  
**Run Name:** `pathmnist_dirichlet_0.1_scaffold_baseline_bn_3` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] SCAFFOLD** achieved **78.68%** accuracy in Round 61 (runtime: 689.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **SCAFFOLD** | `dirichlet_0.1` | 76.74% | **78.68%** | **78.79%** (R74) | 0.7250 | **78.68%** (R61) | R61 | 0.8505 | 689.5s |