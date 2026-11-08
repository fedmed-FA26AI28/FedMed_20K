# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 08:18:12  
**Run Name:** `organamnist_dirichlet_0.1_run_3_seed_44_baseline_fedavg` | **Model:** `baseline` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDAVG** achieved **82.20%** accuracy in Round 80 (runtime: 337.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | `sgd` | No | **FEDAVG** | `dirichlet_0.1` | 82.20% | **82.20%** | **82.86%** (R80) | **82.20%** (R80) | R80 | 0.7098 | 337.7s |