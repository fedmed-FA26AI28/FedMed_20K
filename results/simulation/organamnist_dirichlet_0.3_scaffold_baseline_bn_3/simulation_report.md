# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-04 10:39:33  
**Run Name:** `organamnist_dirichlet_0.3_scaffold_baseline_bn_3` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] SCAFFOLD** achieved **86.97%** accuracy in Round 76 (runtime: 267.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **SCAFFOLD** | `dirichlet_0.3` | 86.65% | **86.97%** | **87.81%** (R41) | 0.4751 | **86.97%** (R76) | R76 | 0.5638 | 267.6s |