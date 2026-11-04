# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-04 12:40:49  
**Run Name:** `pathmnist_dirichlet_0.3_fedbns_baseline_bn_4` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDBNS** achieved **82.73%** accuracy in Round 75 (runtime: 513.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDBNS** | `dirichlet_0.3` | 76.52% | **82.73%** | **79.24%** (R75) | 1.0142 | **82.73%** (R75) | R75 | 1.2874 | 513.0s |