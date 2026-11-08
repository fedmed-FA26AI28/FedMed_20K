# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 10:19:05  
**Run Name:** `organamnist_dirichlet_0.1_run_3_seed_44_baseline_bn_proposed` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] PROPOSED** achieved **94.76%** accuracy in Round 80 (runtime: 288.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | `sgd` | No | **PROPOSED** | `dirichlet_0.1` | 94.76% | **94.76%** | **94.32%** (R80) | **94.76%** (R80) | R80 | 0.2024 | 288.2s |