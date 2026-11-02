# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 12:04:20  
**Run Name:** `bloodmnist_dirichlet_0.1_run_4_seed_45_baseline_bn_fedavg` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDAVG** achieved **87.81%** accuracy in Round 79 (runtime: 110.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | `sgd` | No | **FEDAVG** | `dirichlet_0.1` | 84.39% | **87.81%** | **84.43%** (R75) | **87.81%** (R79) | R79 | 0.4236 | 110.7s |