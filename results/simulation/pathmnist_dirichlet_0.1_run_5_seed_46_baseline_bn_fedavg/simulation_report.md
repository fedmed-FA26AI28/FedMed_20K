# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 13:26:00  
**Run Name:** `pathmnist_dirichlet_0.1_run_5_seed_46_baseline_bn_fedavg` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDAVG** achieved **73.70%** accuracy in Round 47 (runtime: 647.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | `sgd` | No | **FEDAVG** | `dirichlet_0.1` | 70.75% | **73.70%** | **67.58%** (R58) | **73.70%** (R47) | R47 | 1.0160 | 647.5s |