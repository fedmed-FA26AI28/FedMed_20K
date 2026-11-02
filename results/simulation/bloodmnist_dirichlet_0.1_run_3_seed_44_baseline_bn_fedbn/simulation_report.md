# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 11:57:46  
**Run Name:** `bloodmnist_dirichlet_0.1_run_3_seed_44_baseline_bn_fedbn` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDBN** achieved **96.76%** accuracy in Round 79 (runtime: 103.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | `sgd` | No | **FEDBN** | `dirichlet_0.1` | 96.73% | **96.76%** | **88.55%** (R79) | **96.76%** (R79) | R79 | 0.1198 | 103.4s |