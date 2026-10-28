# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-28 17:44:34  
**Run Name:** `organamnist_fedavg_baseline_3` | **Model:** `baseline` | **Seed:** `44` | **Rounds:** 5 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = practical:** **[BEST] FEDAVG** achieved **67.22%** accuracy in Round 5 (runtime: 20.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | 571,707 | 2233.2 KB | `sgd` | No | **FEDAVG** | `practical` | 67.22% | **67.22%** | **67.86%** (R5) | 0.9357 | **67.22%** (R5) | R5 | 0.9409 | 20.0s |