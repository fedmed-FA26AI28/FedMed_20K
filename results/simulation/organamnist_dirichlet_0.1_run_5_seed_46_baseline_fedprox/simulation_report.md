# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 08:50:36  
**Run Name:** `organamnist_dirichlet_0.1_run_5_seed_46_baseline_fedprox` | **Model:** `baseline` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDPROX** achieved **83.20%** accuracy in Round 74 (runtime: 442.1s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | `sgd` | No | **FEDPROX** | `dirichlet_0.1` | 83.16% | **83.20%** | **84.59%** (R78) | **83.20%** (R74) | R74 | 0.8366 | 442.1s |