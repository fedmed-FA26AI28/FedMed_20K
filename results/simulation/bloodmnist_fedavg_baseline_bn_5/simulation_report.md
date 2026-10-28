# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-28 21:17:00  
**Run Name:** `bloodmnist_fedavg_baseline_bn_5` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = practical:** **[BEST] FEDAVG** achieved **86.99%** accuracy in Round 70 (runtime: 94.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDAVG** | `practical` | 85.36% | **86.99%** | **85.02%** (R72) | 0.3774 | **86.99%** (R70) | R70 | 0.3847 | 94.9s |