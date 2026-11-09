# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-09 10:54:42  
**Run Name:** `organamnist_dirichlet_0.1_fedswitch_baseline_bn_5` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSWITCH** achieved **85.34%** accuracy in Round 72 (runtime: 366.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDSWITCH** | `dirichlet_0.1` | 85.23% | **85.34%** | **86.69%** (R72) | 0.4984 | **85.34%** (R72) | R72 | 0.5923 | 366.8s |