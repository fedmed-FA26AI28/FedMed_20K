# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 12:37:04  
**Run Name:** `pathmnist_dirichlet_0.3_fedavg_baseline_bn_5` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDAVG** achieved **81.69%** accuracy in Round 64 (runtime: 500.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDAVG** | `dirichlet_0.3` | 75.07% | **81.69%** | **81.91%** (R70) | 0.9047 | **81.69%** (R64) | R64 | 1.3238 | 500.3s |