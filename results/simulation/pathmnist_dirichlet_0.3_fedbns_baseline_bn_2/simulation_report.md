# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-04 12:22:58  
**Run Name:** `pathmnist_dirichlet_0.3_fedbns_baseline_bn_2` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDBNS** achieved **85.81%** accuracy in Round 73 (runtime: 579.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDBNS** | `dirichlet_0.3` | 85.19% | **85.81%** | **83.92%** (R79) | 0.9158 | **85.81%** (R73) | R73 | 0.8420 | 579.8s |