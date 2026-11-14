# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 16:45:41  
**Run Name:** `bloodmnist_dirichlet_0.3_scaffold_baseline_bn_1` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] SCAFFOLD** achieved **93.10%** accuracy in Round 75 (runtime: 162.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **SCAFFOLD** | `dirichlet_0.3` | 91.96% | **93.10%** | **93.57%** (R75) | 0.1941 | **93.10%** (R75) | R75 | 0.2321 | 162.8s |