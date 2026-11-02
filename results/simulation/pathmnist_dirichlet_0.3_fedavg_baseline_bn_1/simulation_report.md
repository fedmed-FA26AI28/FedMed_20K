# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 12:02:03  
**Run Name:** `pathmnist_dirichlet_0.3_fedavg_baseline_bn_1` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDAVG** achieved **84.22%** accuracy in Round 80 (runtime: 483.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDAVG** | `dirichlet_0.3` | 84.22% | **84.22%** | **83.89%** (R80) | 0.9794 | **84.22%** (R80) | R80 | 0.9191 | 483.7s |