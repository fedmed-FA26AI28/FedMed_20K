# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 09:00:31  
**Run Name:** `organamnist_dirichlet_0.3_fedprox_baseline_2` | **Model:** `baseline` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDPROX** achieved **84.06%** accuracy in Round 73 (runtime: 217.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | 571,707 | 2233.2 KB | `sgd` | No | **FEDPROX** | `dirichlet_0.3` | 84.03% | **84.06%** | **84.14%** (R73) | 0.8085 | **84.06%** (R73) | R73 | 0.8233 | 217.5s |