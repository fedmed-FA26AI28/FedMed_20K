# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 08:05:47  
**Run Name:** `organamnist_dirichlet_0.1_run_2_seed_43_baseline_fedavg` | **Model:** `baseline` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDAVG** achieved **82.42%** accuracy in Round 76 (runtime: 366.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | `sgd` | No | **FEDAVG** | `dirichlet_0.1` | 82.42% | **82.42%** | **82.59%** (R74) | **82.42%** (R76) | R76 | 0.6668 | 366.7s |