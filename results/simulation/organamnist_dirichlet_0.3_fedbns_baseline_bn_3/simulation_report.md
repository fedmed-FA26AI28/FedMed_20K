# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-04 11:03:17  
**Run Name:** `organamnist_dirichlet_0.3_fedbns_baseline_bn_3` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDBNS** achieved **90.67%** accuracy in Round 71 (runtime: 265.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDBNS** | `dirichlet_0.3` | 90.08% | **90.67%** | **91.42%** (R71) | 0.3720 | **90.67%** (R71) | R71 | 0.4382 | 265.4s |