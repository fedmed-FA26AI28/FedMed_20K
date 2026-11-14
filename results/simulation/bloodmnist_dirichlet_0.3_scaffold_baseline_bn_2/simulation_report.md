# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 16:48:53  
**Run Name:** `bloodmnist_dirichlet_0.3_scaffold_baseline_bn_2` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] SCAFFOLD** achieved **92.17%** accuracy in Round 76 (runtime: 176.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **SCAFFOLD** | `dirichlet_0.3` | 90.88% | **92.17%** | **93.06%** (R69) | 0.2041 | **92.17%** (R76) | R76 | 0.2499 | 176.8s |