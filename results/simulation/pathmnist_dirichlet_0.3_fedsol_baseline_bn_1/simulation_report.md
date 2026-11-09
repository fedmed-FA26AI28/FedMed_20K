# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-09 17:56:26  
**Run Name:** `pathmnist_dirichlet_0.3_fedsol_baseline_bn_1` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSOL** achieved **81.62%** accuracy in Round 68 (runtime: 798.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDSOL** | `dirichlet_0.3` | 79.48% | **81.62%** | **81.07%** (R68) | 1.0493 | **81.62%** (R68) | R68 | 1.1781 | 798.9s |