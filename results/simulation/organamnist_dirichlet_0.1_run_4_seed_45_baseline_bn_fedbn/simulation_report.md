# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 10:35:15  
**Run Name:** `organamnist_dirichlet_0.1_run_4_seed_45_baseline_bn_fedbn` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDBN** achieved **91.55%** accuracy in Round 73 (runtime: 245.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | `sgd` | No | **FEDBN** | `dirichlet_0.1` | 91.51% | **91.55%** | **91.31%** (R73) | **91.55%** (R73) | R73 | 0.3314 | 245.8s |