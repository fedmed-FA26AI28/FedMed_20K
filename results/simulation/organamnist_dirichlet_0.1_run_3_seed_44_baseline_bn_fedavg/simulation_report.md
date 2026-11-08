# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 09:56:50  
**Run Name:** `organamnist_dirichlet_0.1_run_3_seed_44_baseline_bn_fedavg` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDAVG** achieved **85.08%** accuracy in Round 75 (runtime: 349.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | `sgd` | No | **FEDAVG** | `dirichlet_0.1` | 84.89% | **85.08%** | **84.88%** (R75) | **85.08%** (R75) | R75 | 0.4722 | 349.7s |