# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 11:51:23  
**Run Name:** `bloodmnist_dirichlet_0.1_run_2_seed_43_baseline_bn_proposed` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] PROPOSED** achieved **97.81%** accuracy in Round 69 (runtime: 133.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | `sgd` | No | **PROPOSED** | `dirichlet_0.1` | 97.72% | **97.81%** | **96.69%** (R77) | **97.81%** (R69) | R69 | 0.0804 | 133.8s |