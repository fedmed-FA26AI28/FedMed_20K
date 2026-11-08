# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 10:03:11  
**Run Name:** `organamnist_dirichlet_0.1_run_3_seed_44_baseline_bn_fedprox` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDPROX** achieved **84.71%** accuracy in Round 75 (runtime: 367.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | `sgd` | No | **FEDPROX** | `dirichlet_0.1` | 84.69% | **84.71%** | **84.58%** (R75) | **84.71%** (R75) | R75 | 0.4817 | 367.4s |