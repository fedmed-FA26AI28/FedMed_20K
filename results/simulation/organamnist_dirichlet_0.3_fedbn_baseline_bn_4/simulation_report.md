# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 21:13:27  
**Run Name:** `organamnist_dirichlet_0.3_fedbn_baseline_bn_4` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDBN** achieved **89.05%** accuracy in Round 79 (runtime: 250.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDBN** | `dirichlet_0.3` | 88.16% | **89.05%** | **89.08%** (R79) | 0.4541 | **89.05%** (R79) | R79 | 0.5250 | 250.7s |