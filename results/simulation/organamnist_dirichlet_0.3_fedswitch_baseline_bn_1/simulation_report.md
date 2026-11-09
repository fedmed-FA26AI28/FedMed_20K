# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-09 11:35:15  
**Run Name:** `organamnist_dirichlet_0.3_fedswitch_baseline_bn_1` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSWITCH** achieved **87.78%** accuracy in Round 80 (runtime: 269.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDSWITCH** | `dirichlet_0.3` | 87.78% | **87.78%** | **87.55%** (R71) | 0.5036 | **87.78%** (R80) | R80 | 0.5042 | 269.9s |