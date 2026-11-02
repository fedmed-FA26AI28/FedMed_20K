# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 14:10:31  
**Run Name:** `organamnist_dirichlet_0.3_fedsld_baseline_bn_5` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSLD** achieved **87.61%** accuracy in Round 67 (runtime: 279.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDSLD** | `dirichlet_0.3` | 87.43% | **87.61%** | **87.07%** (R79) | 0.4629 | **87.61%** (R67) | R67 | 0.4433 | 279.0s |