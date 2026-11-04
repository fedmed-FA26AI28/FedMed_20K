# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-04 10:34:51  
**Run Name:** `organamnist_dirichlet_0.3_scaffold_baseline_bn_2` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] SCAFFOLD** achieved **87.51%** accuracy in Round 72 (runtime: 228.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **SCAFFOLD** | `dirichlet_0.3` | 87.46% | **87.51%** | **87.58%** (R62) | 0.4534 | **87.51%** (R72) | R72 | 0.4890 | 228.6s |