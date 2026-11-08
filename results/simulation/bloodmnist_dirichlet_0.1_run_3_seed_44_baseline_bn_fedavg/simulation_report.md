# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 11:53:34  
**Run Name:** `bloodmnist_dirichlet_0.1_run_3_seed_44_baseline_bn_fedavg` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDAVG** achieved **88.45%** accuracy in Round 69 (runtime: 119.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | `sgd` | No | **FEDAVG** | `dirichlet_0.1` | 86.32% | **88.45%** | **82.38%** (R64) | **88.45%** (R69) | R69 | 0.3688 | 119.8s |