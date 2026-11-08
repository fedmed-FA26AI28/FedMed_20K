# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 10:45:50  
**Run Name:** `organamnist_dirichlet_0.1_run_4_seed_45_baseline_bn_proposed` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] PROPOSED** achieved **92.13%** accuracy in Round 79 (runtime: 260.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | `sgd` | No | **PROPOSED** | `dirichlet_0.1` | 92.12% | **92.13%** | **92.23%** (R79) | **92.13%** (R79) | R79 | 0.3067 | 260.8s |