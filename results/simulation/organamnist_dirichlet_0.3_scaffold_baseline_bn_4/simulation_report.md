# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-04 10:44:19  
**Run Name:** `organamnist_dirichlet_0.3_scaffold_baseline_bn_4` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] SCAFFOLD** achieved **87.03%** accuracy in Round 79 (runtime: 271.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **SCAFFOLD** | `dirichlet_0.3` | 86.45% | **87.03%** | **86.63%** (R73) | 0.5210 | **87.03%** (R79) | R79 | 0.5221 | 271.3s |