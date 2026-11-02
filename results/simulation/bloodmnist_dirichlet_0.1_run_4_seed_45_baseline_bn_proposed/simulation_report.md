# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 12:12:24  
**Run Name:** `bloodmnist_dirichlet_0.1_run_4_seed_45_baseline_bn_proposed` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] PROPOSED** achieved **97.14%** accuracy in Round 79 (runtime: 105.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | `sgd` | No | **PROPOSED** | `dirichlet_0.1` | 96.78% | **97.14%** | **95.74%** (R67) | **97.14%** (R79) | R79 | 0.1164 | 105.4s |