# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 08:47:14  
**Run Name:** `organamnist_dirichlet_0.3_fedavg_baseline_4` | **Model:** `baseline` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDAVG** achieved **83.73%** accuracy in Round 70 (runtime: 242.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | 571,707 | 2233.2 KB | `sgd` | No | **FEDAVG** | `dirichlet_0.3` | 83.65% | **83.73%** | **83.11%** (R68) | 0.8559 | **83.73%** (R70) | R70 | 0.8992 | 242.0s |