# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 14:54:33  
**Run Name:** `pathmnist_dirichlet_0.3_fedsld_baseline_bn_5` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSLD** achieved **75.22%** accuracy in Round 80 (runtime: 512.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDSLD** | `dirichlet_0.3` | 75.22% | **75.22%** | **74.74%** (R72) | 0.9346 | **75.22%** (R80) | R80 | 0.9832 | 512.0s |