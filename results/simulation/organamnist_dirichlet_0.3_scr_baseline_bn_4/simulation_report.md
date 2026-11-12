# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-10 14:27:16  
**Run Name:** `dirichlet03_bn_organamnist_s45` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] SCR** achieved **86.91%** accuracy in Round 66 (runtime: 1989.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **SCR** | `dirichlet_0.3` | 86.66% | **86.91%** | **86.66%** (R73) | 0.5226 | **86.91%** (R66) | R66 | 0.5226 | 1989.6s |