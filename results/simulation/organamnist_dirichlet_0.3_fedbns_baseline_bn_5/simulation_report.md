# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-04 11:12:52  
**Run Name:** `organamnist_dirichlet_0.3_fedbns_baseline_bn_5` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDBNS** achieved **91.37%** accuracy in Round 79 (runtime: 285.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDBNS** | `dirichlet_0.3` | 91.33% | **91.37%** | **91.16%** (R79) | 0.3778 | **91.37%** (R79) | R79 | 0.3706 | 285.7s |