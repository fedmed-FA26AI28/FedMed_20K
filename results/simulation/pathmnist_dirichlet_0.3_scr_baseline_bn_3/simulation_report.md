# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-11 05:20:15  
**Run Name:** `dirichlet03_bn_pathmnist_s44` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] SCR** achieved **84.60%** accuracy in Round 48 (runtime: 4242.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **SCR** | `dirichlet_0.3` | 83.88% | **84.60%** | **83.88%** (R48) | 0.6546 | **84.60%** (R48) | R48 | 0.6546 | 4242.2s |