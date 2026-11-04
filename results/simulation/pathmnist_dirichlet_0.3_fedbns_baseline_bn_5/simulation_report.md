# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-04 12:49:50  
**Run Name:** `pathmnist_dirichlet_0.3_fedbns_baseline_bn_5` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDBNS** achieved **80.68%** accuracy in Round 70 (runtime: 527.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDBNS** | `dirichlet_0.3` | 53.87% | **80.68%** | **82.24%** (R70) | 1.0928 | **80.68%** (R70) | R70 | 5.9815 | 527.0s |