# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-09 11:53:14  
**Run Name:** `organamnist_dirichlet_0.3_fedswitch_baseline_bn_5` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSWITCH** achieved **87.38%** accuracy in Round 76 (runtime: 296.1s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDSWITCH** | `dirichlet_0.3` | 87.09% | **87.38%** | **86.82%** (R76) | 0.5158 | **87.38%** (R76) | R76 | 0.5146 | 296.1s |