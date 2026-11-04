# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-04 11:55:03  
**Run Name:** `pathmnist_dirichlet_0.3_scaffold_baseline_bn_4` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] SCAFFOLD** achieved **84.01%** accuracy in Round 42 (runtime: 512.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **SCAFFOLD** | `dirichlet_0.3` | 83.64% | **84.01%** | **80.78%** (R60) | 0.7436 | **84.01%** (R42) | R42 | 0.6497 | 512.7s |