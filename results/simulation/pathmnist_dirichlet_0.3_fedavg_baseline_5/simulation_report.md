# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 10:01:30  
**Run Name:** `pathmnist_dirichlet_0.3_fedavg_baseline_5` | **Model:** `baseline` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDAVG** achieved **79.58%** accuracy in Round 46 (runtime: 476.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | 570,705 | 2229.3 KB | `sgd` | No | **FEDAVG** | `dirichlet_0.3` | 72.90% | **79.58%** | **80.80%** (R46) | 0.9231 | **79.58%** (R46) | R46 | 1.5927 | 476.3s |