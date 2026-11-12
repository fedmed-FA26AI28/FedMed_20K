# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-10 13:53:38  
**Run Name:** `dirichlet03_bn_organamnist_s44` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] SCR** achieved **86.70%** accuracy in Round 78 (runtime: 1984.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **SCR** | `dirichlet_0.3` | 87.46% | **86.70%** | **87.46%** (R71) | 0.5445 | **86.70%** (R78) | R78 | 0.5445 | 1984.9s |