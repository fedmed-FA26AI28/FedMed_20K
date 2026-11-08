# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 08:29:54  
**Run Name:** `organamnist_dirichlet_0.1_run_4_seed_45_baseline_fedavg` | **Model:** `baseline` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDAVG** achieved **82.57%** accuracy in Round 77 (runtime: 325.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | `sgd` | No | **FEDAVG** | `dirichlet_0.1` | 82.32% | **82.57%** | **80.50%** (R77) | **82.57%** (R77) | R77 | 0.7479 | 325.6s |