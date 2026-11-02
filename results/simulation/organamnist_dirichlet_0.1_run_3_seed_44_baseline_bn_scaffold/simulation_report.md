# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 10:14:05  
**Run Name:** `organamnist_dirichlet_0.1_run_3_seed_44_baseline_bn_scaffold` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] SCAFFOLD** achieved **85.07%** accuracy in Round 80 (runtime: 364.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | `sgd` | No | **SCAFFOLD** | `dirichlet_0.1` | 85.07% | **85.07%** | **85.24%** (R80) | **85.07%** (R80) | R80 | 0.4831 | 364.7s |