# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 07:53:05  
**Run Name:** `organamnist_dirichlet_0.1_run_1_seed_42_baseline_fedavg` | **Model:** `baseline` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDAVG** achieved **84.12%** accuracy in Round 78 (runtime: 357.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | `sgd` | No | **FEDAVG** | `dirichlet_0.1` | 84.00% | **84.12%** | **81.14%** (R78) | **84.12%** (R78) | R78 | 0.6788 | 357.6s |