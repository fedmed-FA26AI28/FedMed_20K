# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 10:41:17  
**Run Name:** `organamnist_dirichlet_0.1_run_4_seed_45_baseline_bn_scaffold` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] SCAFFOLD** achieved **85.92%** accuracy in Round 77 (runtime: 348.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | `sgd` | No | **SCAFFOLD** | `dirichlet_0.1` | 85.58% | **85.92%** | **83.73%** (R64) | **85.92%** (R77) | R77 | 0.5124 | 348.6s |