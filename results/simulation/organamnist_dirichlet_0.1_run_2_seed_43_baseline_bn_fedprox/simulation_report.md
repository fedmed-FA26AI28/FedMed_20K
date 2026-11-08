# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 09:33:32  
**Run Name:** `organamnist_dirichlet_0.1_run_2_seed_43_baseline_bn_fedprox` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDPROX** achieved **84.13%** accuracy in Round 79 (runtime: 396.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | `sgd` | No | **FEDPROX** | `dirichlet_0.1` | 83.82% | **84.13%** | **83.69%** (R79) | **84.13%** (R79) | R79 | 0.5289 | 396.2s |