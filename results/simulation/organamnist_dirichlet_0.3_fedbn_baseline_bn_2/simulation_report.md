# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 21:04:29  
**Run Name:** `organamnist_dirichlet_0.3_fedbn_baseline_bn_2` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDBN** achieved **90.74%** accuracy in Round 76 (runtime: 220.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDBN** | `dirichlet_0.3` | 88.87% | **90.74%** | **90.76%** (R76) | 0.3630 | **90.74%** (R76) | R76 | 0.4374 | 220.8s |