# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 08:35:02  
**Run Name:** `organamnist_dirichlet_0.3_fedavg_baseline_1` | **Model:** `baseline` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDAVG** achieved **83.06%** accuracy in Round 49 (runtime: 252.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | 571,707 | 2233.2 KB | `sgd` | No | **FEDAVG** | `dirichlet_0.3` | 82.93% | **83.06%** | **81.95%** (R67) | 0.9115 | **83.06%** (R49) | R49 | 0.8860 | 252.3s |