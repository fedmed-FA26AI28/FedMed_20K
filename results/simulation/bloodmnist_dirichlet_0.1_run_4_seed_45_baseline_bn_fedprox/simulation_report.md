# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 12:06:27  
**Run Name:** `bloodmnist_dirichlet_0.1_run_4_seed_45_baseline_bn_fedprox` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDPROX** achieved **86.67%** accuracy in Round 77 (runtime: 115.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | `sgd` | No | **FEDPROX** | `dirichlet_0.1` | 78.46% | **86.67%** | **84.06%** (R79) | **86.67%** (R77) | R77 | 0.6903 | 115.0s |