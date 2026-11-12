# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-11 03:26:08  
**Run Name:** `dirichlet03_bn_pathmnist_s43` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] SCR** achieved **84.30%** accuracy in Round 37 (runtime: 3923.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **SCR** | `dirichlet_0.3` | 82.76% | **84.30%** | **82.76%** (R46) | 0.7386 | **84.30%** (R37) | R37 | 0.7386 | 3923.2s |