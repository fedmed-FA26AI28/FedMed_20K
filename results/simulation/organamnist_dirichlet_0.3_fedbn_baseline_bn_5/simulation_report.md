# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 21:18:22  
**Run Name:** `organamnist_dirichlet_0.3_fedbn_baseline_bn_5` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDBN** achieved **91.06%** accuracy in Round 80 (runtime: 280.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDBN** | `dirichlet_0.3` | 91.06% | **91.06%** | **90.83%** (R79) | 0.3881 | **91.06%** (R80) | R80 | 0.3791 | 280.5s |