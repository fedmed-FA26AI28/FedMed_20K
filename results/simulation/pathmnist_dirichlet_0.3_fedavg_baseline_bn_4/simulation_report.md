# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 12:28:29  
**Run Name:** `pathmnist_dirichlet_0.3_fedavg_baseline_bn_4` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDAVG** achieved **84.37%** accuracy in Round 48 (runtime: 475.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDAVG** | `dirichlet_0.3` | 82.77% | **84.37%** | **80.91%** (R60) | 0.8706 | **84.37%** (R48) | R48 | 0.7639 | 475.0s |