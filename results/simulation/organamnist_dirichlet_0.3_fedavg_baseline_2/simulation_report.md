# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 08:38:42  
**Run Name:** `organamnist_dirichlet_0.3_fedavg_baseline_2` | **Model:** `baseline` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDAVG** achieved **84.05%** accuracy in Round 73 (runtime: 205.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | 571,707 | 2233.2 KB | `sgd` | No | **FEDAVG** | `dirichlet_0.3` | 84.01% | **84.05%** | **84.13%** (R73) | 0.8105 | **84.05%** (R73) | R73 | 0.8262 | 205.4s |