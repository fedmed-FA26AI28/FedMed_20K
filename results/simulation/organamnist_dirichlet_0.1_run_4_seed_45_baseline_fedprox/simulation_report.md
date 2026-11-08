# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 08:35:40  
**Run Name:** `organamnist_dirichlet_0.1_run_4_seed_45_baseline_fedprox` | **Model:** `baseline` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDPROX** achieved **82.24%** accuracy in Round 77 (runtime: 333.1s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | `sgd` | No | **FEDPROX** | `dirichlet_0.1` | 82.21% | **82.24%** | **80.31%** (R80) | **82.24%** (R77) | R77 | 0.7514 | 333.1s |