# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-04 12:13:04  
**Run Name:** `pathmnist_dirichlet_0.3_fedbns_baseline_bn_1` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDBNS** achieved **86.81%** accuracy in Round 79 (runtime: 515.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDBNS** | `dirichlet_0.3` | 84.69% | **86.81%** | **85.41%** (R79) | 0.9619 | **86.81%** (R79) | R79 | 0.9871 | 515.5s |