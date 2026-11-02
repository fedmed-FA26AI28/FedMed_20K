# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 21:09:02  
**Run Name:** `organamnist_dirichlet_0.3_fedbn_baseline_bn_3` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDBN** achieved **90.13%** accuracy in Round 78 (runtime: 257.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDBN** | `dirichlet_0.3` | 88.59% | **90.13%** | **90.89%** (R70) | 0.3953 | **90.13%** (R78) | R78 | 0.4793 | 257.9s |