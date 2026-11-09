# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-09 10:48:20  
**Run Name:** `organamnist_dirichlet_0.1_fedswitch_baseline_bn_4` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSWITCH** achieved **85.85%** accuracy in Round 62 (runtime: 255.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDSWITCH** | `dirichlet_0.1` | 85.70% | **85.85%** | **83.81%** (R65) | 0.5536 | **85.85%** (R62) | R62 | 0.5130 | 255.0s |