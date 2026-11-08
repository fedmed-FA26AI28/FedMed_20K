# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 13:50:33  
**Run Name:** `pathmnist_dirichlet_0.1_run_2_seed_43_baseline_bn_fedbn` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDBN** achieved **89.78%** accuracy in Round 75 (runtime: 669.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | `sgd` | No | **FEDBN** | `dirichlet_0.1` | 88.83% | **89.78%** | **88.57%** (R75) | **89.78%** (R75) | R75 | 0.5778 | 669.3s |