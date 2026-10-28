# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-28 17:43:21  
**Run Name:** `organamnist_fedavg_baseline_1` | **Model:** `baseline` | **Seed:** `42` | **Rounds:** 5 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = practical:** **[BEST] FEDAVG** achieved **66.93%** accuracy in Round 5 (runtime: 23.1s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | 571,707 | 2233.2 KB | `sgd` | No | **FEDAVG** | `practical` | 66.93% | **66.93%** | **64.53%** (R5) | 1.0595 | **66.93%** (R5) | R5 | 0.9626 | 23.1s |