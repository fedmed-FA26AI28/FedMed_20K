# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 09:03:23  
**Run Name:** `organamnist_dirichlet_0.1_run_1_seed_42_baseline_bn_fedprox` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDPROX** achieved **84.42%** accuracy in Round 60 (runtime: 380.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | `sgd` | No | **FEDPROX** | `dirichlet_0.1` | 80.85% | **84.42%** | **79.81%** (R78) | **84.42%** (R60) | R60 | 0.5989 | 380.6s |