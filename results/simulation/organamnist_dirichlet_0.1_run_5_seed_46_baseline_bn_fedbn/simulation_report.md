# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 11:07:22  
**Run Name:** `organamnist_dirichlet_0.1_run_5_seed_46_baseline_bn_fedbn` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDBN** achieved **91.23%** accuracy in Round 79 (runtime: 351.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | `sgd` | No | **FEDBN** | `dirichlet_0.1` | 91.06% | **91.23%** | **92.98%** (R79) | **91.23%** (R79) | R79 | 0.4122 | 351.9s |