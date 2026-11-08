# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 14:01:46  
**Run Name:** `pathmnist_dirichlet_0.1_run_3_seed_44_baseline_bn_fedbn` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDBN** achieved **89.26%** accuracy in Round 77 (runtime: 660.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | `sgd` | No | **FEDBN** | `dirichlet_0.1` | 87.81% | **89.26%** | **90.08%** (R77) | **89.26%** (R77) | R77 | 0.5927 | 660.5s |