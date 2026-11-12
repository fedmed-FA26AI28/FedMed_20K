# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-10 13:20:06  
**Run Name:** `dirichlet03_bn_organamnist_s43` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] SCR** achieved **87.41%** accuracy in Round 79 (runtime: 1958.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **SCR** | `dirichlet_0.3` | 87.58% | **87.41%** | **87.58%** (R80) | 0.4760 | **87.41%** (R79) | R79 | 0.4760 | 1958.8s |