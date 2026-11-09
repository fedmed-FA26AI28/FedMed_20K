# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-09 10:43:50  
**Run Name:** `organamnist_dirichlet_0.1_fedswitch_baseline_bn_3` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSWITCH** achieved **84.01%** accuracy in Round 79 (runtime: 271.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDSWITCH** | `dirichlet_0.1` | 83.88% | **84.01%** | **84.17%** (R77) | 0.5060 | **84.01%** (R79) | R79 | 0.5349 | 271.6s |