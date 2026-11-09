# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-09 11:43:41  
**Run Name:** `organamnist_dirichlet_0.3_fedswitch_baseline_bn_3` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSWITCH** achieved **86.96%** accuracy in Round 76 (runtime: 258.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDSWITCH** | `dirichlet_0.3` | 86.40% | **86.96%** | **87.93%** (R76) | 0.5020 | **86.96%** (R76) | R76 | 0.5426 | 258.2s |