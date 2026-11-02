# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 12:11:30  
**Run Name:** `pathmnist_dirichlet_0.3_fedavg_baseline_bn_2` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDAVG** achieved **84.44%** accuracy in Round 44 (runtime: 552.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDAVG** | `dirichlet_0.3` | 83.31% | **84.44%** | **83.15%** (R44) | 0.7696 | **84.44%** (R44) | R44 | 0.8583 | 552.9s |