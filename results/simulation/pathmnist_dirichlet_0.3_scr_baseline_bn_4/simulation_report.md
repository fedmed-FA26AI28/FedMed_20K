# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-11 06:30:48  
**Run Name:** `dirichlet03_bn_pathmnist_s45` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] SCR** achieved **84.04%** accuracy in Round 79 (runtime: 4203.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **SCR** | `dirichlet_0.3` | 80.49% | **84.04%** | **80.49%** (R79) | 0.8792 | **84.04%** (R79) | R79 | 0.8792 | 4203.6s |