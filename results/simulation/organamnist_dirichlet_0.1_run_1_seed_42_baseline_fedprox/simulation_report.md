# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 07:59:27  
**Run Name:** `organamnist_dirichlet_0.1_run_1_seed_42_baseline_fedprox` | **Model:** `baseline` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDPROX** achieved **83.77%** accuracy in Round 78 (runtime: 369.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | `sgd` | No | **FEDPROX** | `dirichlet_0.1` | 83.70% | **83.77%** | **80.94%** (R80) | **83.77%** (R78) | R78 | 0.6840 | 369.2s |