# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-09 18:38:52  
**Run Name:** `pathmnist_dirichlet_0.3_fedsol_baseline_bn_4` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSOL** achieved **81.28%** accuracy in Round 66 (runtime: 795.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDSOL** | `dirichlet_0.3` | 80.47% | **81.28%** | **78.08%** (R56) | 1.1576 | **81.28%** (R66) | R66 | 0.9475 | 795.3s |