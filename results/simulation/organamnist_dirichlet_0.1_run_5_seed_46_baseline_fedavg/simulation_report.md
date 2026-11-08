# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 08:43:01  
**Run Name:** `organamnist_dirichlet_0.1_run_5_seed_46_baseline_fedavg` | **Model:** `baseline` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDAVG** achieved **83.38%** accuracy in Round 80 (runtime: 427.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | `sgd` | No | **FEDAVG** | `dirichlet_0.1` | 83.38% | **83.38%** | **84.58%** (R78) | **83.38%** (R80) | R80 | 0.8308 | 427.6s |