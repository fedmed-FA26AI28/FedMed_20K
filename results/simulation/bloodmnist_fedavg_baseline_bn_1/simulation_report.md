# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-28 21:08:48  
**Run Name:** `bloodmnist_fedavg_baseline_bn_1` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = practical:** **[BEST] FEDAVG** achieved **91.14%** accuracy in Round 80 (runtime: 125.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDAVG** | `practical` | 91.14% | **91.14%** | **90.03%** (R78) | 0.2808 | **91.14%** (R80) | R80 | 0.2454 | 125.7s |