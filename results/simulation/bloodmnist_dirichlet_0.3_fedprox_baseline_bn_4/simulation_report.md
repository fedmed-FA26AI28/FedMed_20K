# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 11:05:43  
**Run Name:** `bloodmnist_dirichlet_0.3_fedprox_baseline_bn_4` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDPROX** achieved **92.46%** accuracy in Round 76 (runtime: 105.1s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDPROX** | `dirichlet_0.3` | 92.34% | **92.46%** | **92.25%** (R77) | 0.2180 | **92.46%** (R76) | R76 | 0.2239 | 105.1s |