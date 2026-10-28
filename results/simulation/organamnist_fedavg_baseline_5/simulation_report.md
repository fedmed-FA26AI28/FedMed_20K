# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-28 18:06:59  
**Run Name:** `organamnist_fedavg_baseline_5` | **Model:** `baseline` | **Seed:** `42` | **Rounds:** 1 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = practical:** **[BEST] FEDAVG** achieved **35.03%** accuracy in Round 1 (runtime: 8.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | 571,707 | 2233.2 KB | `sgd` | No | **FEDAVG** | `practical` | 35.03% | **35.03%** | **28.95%** (R1) | 2.0891 | **35.03%** (R1) | R1 | 1.9935 | 8.2s |