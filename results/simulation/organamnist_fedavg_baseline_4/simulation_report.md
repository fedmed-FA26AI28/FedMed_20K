# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-28 17:45:11  
**Run Name:** `organamnist_fedavg_baseline_4` | **Model:** `baseline` | **Seed:** `45` | **Rounds:** 5 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = practical:** **[BEST] FEDAVG** achieved **65.46%** accuracy in Round 5 (runtime: 20.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | 571,707 | 2233.2 KB | `sgd` | No | **FEDAVG** | `practical` | 65.46% | **65.46%** | **63.27%** (R5) | 1.0361 | **65.46%** (R5) | R5 | 0.9745 | 20.5s |