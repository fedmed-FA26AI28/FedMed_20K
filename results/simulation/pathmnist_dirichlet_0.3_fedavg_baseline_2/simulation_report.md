# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 09:37:19  
**Run Name:** `pathmnist_dirichlet_0.3_fedavg_baseline_2` | **Model:** `baseline` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDAVG** achieved **78.22%** accuracy in Round 67 (runtime: 545.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | 570,705 | 2229.3 KB | `sgd` | No | **FEDAVG** | `dirichlet_0.3` | 77.33% | **78.22%** | **76.56%** (R67) | 1.8169 | **78.22%** (R67) | R67 | 1.9361 | 545.7s |