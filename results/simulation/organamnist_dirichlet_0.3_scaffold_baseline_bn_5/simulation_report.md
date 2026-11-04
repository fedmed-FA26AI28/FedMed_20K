# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-04 10:49:21  
**Run Name:** `organamnist_dirichlet_0.3_scaffold_baseline_bn_5` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] SCAFFOLD** achieved **87.49%** accuracy in Round 63 (runtime: 288.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **SCAFFOLD** | `dirichlet_0.3` | 87.37% | **87.49%** | **86.99%** (R63) | 0.5088 | **87.49%** (R63) | R63 | 0.5182 | 288.7s |