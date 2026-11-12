# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-11 01:14:43  
**Run Name:** `dirichlet03_bn_bloodmnist_s46` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] SCR** achieved **92.40%** accuracy in Round 67 (runtime: 630.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **SCR** | `dirichlet_0.3` | 92.83% | **92.40%** | **92.83%** (R67) | 0.2244 | **92.40%** (R67) | R67 | 0.2244 | 630.0s |