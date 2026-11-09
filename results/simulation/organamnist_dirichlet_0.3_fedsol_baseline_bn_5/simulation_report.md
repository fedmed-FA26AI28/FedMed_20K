# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-09 12:25:32  
**Run Name:** `organamnist_dirichlet_0.3_fedsol_baseline_bn_5` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSOL** achieved **88.08%** accuracy in Round 77 (runtime: 400.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDSOL** | `dirichlet_0.3` | 87.78% | **88.08%** | **87.57%** (R77) | 0.4574 | **88.08%** (R77) | R77 | 0.4595 | 400.5s |