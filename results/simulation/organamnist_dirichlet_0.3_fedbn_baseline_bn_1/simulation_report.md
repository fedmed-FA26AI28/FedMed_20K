# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 21:00:35  
**Run Name:** `organamnist_dirichlet_0.3_fedbn_baseline_bn_1` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDBN** achieved **91.06%** accuracy in Round 79 (runtime: 337.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDBN** | `dirichlet_0.3` | 90.94% | **91.06%** | **91.09%** (R75) | 0.3781 | **91.06%** (R79) | R79 | 0.3845 | 337.7s |