# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 08:52:00  
**Run Name:** `organamnist_dirichlet_0.3_fedavg_baseline_5` | **Model:** `baseline` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDAVG** achieved **83.59%** accuracy in Round 71 (runtime: 271.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | 571,707 | 2233.2 KB | `sgd` | No | **FEDAVG** | `dirichlet_0.3` | 83.53% | **83.59%** | **82.97%** (R71) | 0.8431 | **83.59%** (R71) | R71 | 0.8408 | 271.0s |