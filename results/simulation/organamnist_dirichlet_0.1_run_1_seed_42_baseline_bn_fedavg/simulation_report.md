# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 08:56:49  
**Run Name:** `organamnist_dirichlet_0.1_run_1_seed_42_baseline_bn_fedavg` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDAVG** achieved **84.78%** accuracy in Round 68 (runtime: 360.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | `sgd` | No | **FEDAVG** | `dirichlet_0.1` | 78.99% | **84.78%** | **80.21%** (R68) | **84.78%** (R68) | R68 | 0.6673 | 360.4s |