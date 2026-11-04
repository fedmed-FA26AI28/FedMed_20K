# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-04 11:07:52  
**Run Name:** `organamnist_dirichlet_0.3_fedbns_baseline_bn_4` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDBNS** achieved **89.16%** accuracy in Round 72 (runtime: 260.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDBNS** | `dirichlet_0.3` | 88.44% | **89.16%** | **89.34%** (R72) | 0.4338 | **89.16%** (R72) | R72 | 0.5268 | 260.0s |