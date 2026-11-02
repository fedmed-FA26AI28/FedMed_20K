# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 12:20:20  
**Run Name:** `pathmnist_dirichlet_0.3_fedavg_baseline_bn_3` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDAVG** achieved **83.94%** accuracy in Round 39 (runtime: 516.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDAVG** | `dirichlet_0.3` | 83.66% | **83.94%** | **83.46%** (R39) | 0.6804 | **83.94%** (R39) | R39 | 0.7779 | 516.2s |