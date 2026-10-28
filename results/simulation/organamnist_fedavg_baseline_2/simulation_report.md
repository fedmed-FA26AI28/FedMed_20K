# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-28 17:43:59  
**Run Name:** `organamnist_fedavg_baseline_2` | **Model:** `baseline` | **Seed:** `43` | **Rounds:** 5 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = practical:** **[BEST] FEDAVG** achieved **66.09%** accuracy in Round 5 (runtime: 22.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | 571,707 | 2233.2 KB | `sgd` | No | **FEDAVG** | `practical` | 66.09% | **66.09%** | **63.71%** (R5) | 1.0433 | **66.09%** (R5) | R5 | 0.9699 | 22.9s |