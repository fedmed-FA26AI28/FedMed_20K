# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 10:30:56  
**Run Name:** `organamnist_dirichlet_0.1_run_4_seed_45_baseline_bn_fedprox` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDPROX** achieved **85.88%** accuracy in Round 76 (runtime: 350.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | `sgd` | No | **FEDPROX** | `dirichlet_0.1` | 85.61% | **85.88%** | **83.71%** (R76) | **85.88%** (R76) | R76 | 0.4964 | 350.9s |