# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 17:13:30  
**Run Name:** `bloodmnist_dirichlet_0.3_fedbns_baseline_bn_5` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDBNS** achieved **94.71%** accuracy in Round 71 (runtime: 174.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDBNS** | `dirichlet_0.3` | 94.01% | **94.71%** | **95.20%** (R71) | 0.1651 | **94.71%** (R71) | R71 | 0.2072 | 174.6s |