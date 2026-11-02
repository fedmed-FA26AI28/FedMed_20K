# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 12:10:27  
**Run Name:** `bloodmnist_dirichlet_0.1_run_4_seed_45_baseline_bn_scaffold` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] SCAFFOLD** achieved **86.96%** accuracy in Round 79 (runtime: 122.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | `sgd` | No | **SCAFFOLD** | `dirichlet_0.1` | 84.71% | **86.96%** | **84.94%** (R79) | **86.96%** (R79) | R79 | 0.4342 | 122.7s |