# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 13:14:59  
**Run Name:** `pathmnist_dirichlet_0.1_run_4_seed_45_baseline_bn_fedavg` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDAVG** achieved **69.08%** accuracy in Round 73 (runtime: 641.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | `sgd` | No | **FEDAVG** | `dirichlet_0.1` | 63.09% | **69.08%** | **67.39%** (R73) | **69.08%** (R73) | R73 | 1.4608 | 641.5s |