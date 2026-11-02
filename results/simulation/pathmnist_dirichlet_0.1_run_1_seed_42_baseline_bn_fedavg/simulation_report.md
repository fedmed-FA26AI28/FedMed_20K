# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 12:40:27  
**Run Name:** `pathmnist_dirichlet_0.1_run_1_seed_42_baseline_bn_fedavg` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDAVG** achieved **72.74%** accuracy in Round 79 (runtime: 693.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | `sgd` | No | **FEDAVG** | `dirichlet_0.1` | 64.48% | **72.74%** | **69.46%** (R77) | **72.74%** (R79) | R79 | 1.2210 | 693.8s |