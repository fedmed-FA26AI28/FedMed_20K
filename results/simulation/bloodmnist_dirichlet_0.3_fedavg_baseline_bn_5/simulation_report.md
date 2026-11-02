# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 10:58:06  
**Run Name:** `bloodmnist_dirichlet_0.3_fedavg_baseline_bn_5` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDAVG** achieved **93.16%** accuracy in Round 53 (runtime: 96.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDAVG** | `dirichlet_0.3` | 91.93% | **93.16%** | **93.72%** (R53) | 0.1918 | **93.16%** (R53) | R53 | 0.2449 | 96.7s |