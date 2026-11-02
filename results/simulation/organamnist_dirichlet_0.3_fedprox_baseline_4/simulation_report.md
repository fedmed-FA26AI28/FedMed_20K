# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 09:09:21  
**Run Name:** `organamnist_dirichlet_0.3_fedprox_baseline_4` | **Model:** `baseline` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDPROX** achieved **83.77%** accuracy in Round 77 (runtime: 246.1s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | 571,707 | 2233.2 KB | `sgd` | No | **FEDPROX** | `dirichlet_0.3` | 83.71% | **83.77%** | **83.63%** (R29) | 0.6978 | **83.77%** (R77) | R77 | 0.9048 | 246.1s |