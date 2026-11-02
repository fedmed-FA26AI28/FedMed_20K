# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 10:56:14  
**Run Name:** `bloodmnist_dirichlet_0.3_fedavg_baseline_bn_4` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDAVG** achieved **92.81%** accuracy in Round 76 (runtime: 98.1s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDAVG** | `dirichlet_0.3` | 92.69% | **92.81%** | **92.58%** (R76) | 0.2207 | **92.81%** (R76) | R76 | 0.2219 | 98.1s |