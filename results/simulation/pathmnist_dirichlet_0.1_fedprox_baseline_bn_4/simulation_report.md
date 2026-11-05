# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-05 15:56:26  
**Run Name:** `pathmnist_dirichlet_0.1_fedprox_baseline_bn_4` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDPROX** achieved **71.64%** accuracy in Round 74 (runtime: 663.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDPROX** | `dirichlet_0.1` | 66.89% | **71.64%** | **68.84%** (R74) | 1.3516 | **71.64%** (R74) | R74 | 1.2234 | 663.0s |