# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 08:24:16  
**Run Name:** `organamnist_dirichlet_0.1_run_3_seed_44_baseline_fedprox` | **Model:** `baseline` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDPROX** achieved **82.14%** accuracy in Round 72 (runtime: 351.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | `sgd` | No | **FEDPROX** | `dirichlet_0.1` | 82.07% | **82.14%** | **82.68%** (R68) | **82.14%** (R72) | R72 | 0.7049 | 351.2s |