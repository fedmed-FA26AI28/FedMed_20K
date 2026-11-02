# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 11:55:51  
**Run Name:** `bloodmnist_dirichlet_0.1_run_3_seed_44_baseline_bn_fedprox` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDPROX** achieved **88.34%** accuracy in Round 76 (runtime: 125.1s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | `sgd` | No | **FEDPROX** | `dirichlet_0.1` | 85.94% | **88.34%** | **82.69%** (R76) | **88.34%** (R76) | R76 | 0.3757 | 125.1s |