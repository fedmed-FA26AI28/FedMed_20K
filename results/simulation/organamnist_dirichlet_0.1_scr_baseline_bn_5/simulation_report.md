# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-04 15:34:56  
**Run Name:** `dirichlet01_bn_organamnist_s46` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] SCR** achieved **84.99%** accuracy in Round 72 (runtime: 2180.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **SCR** | `dirichlet_0.1` | 84.84% | **84.99%** | **85.99%** (R72) | 0.5272 | **84.99%** (R72) | R72 | 0.6022 | 2180.6s |