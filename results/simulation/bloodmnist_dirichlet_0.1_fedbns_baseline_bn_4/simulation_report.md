# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 15:03:17  
**Run Name:** `bloodmnist_dirichlet_0.1_fedbns_baseline_bn_4` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDBNS** achieved **96.99%** accuracy in Round 79 (runtime: 173.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDBNS** | `dirichlet_0.1` | 95.47% | **96.99%** | **95.13%** (R79) | 0.1503 | **96.99%** (R79) | R79 | 0.1845 | 173.6s |