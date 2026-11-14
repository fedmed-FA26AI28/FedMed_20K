# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 13:20:42  
**Run Name:** `bloodmnist_practical_fedswitch_baseline_bn_3` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = practical:** **[BEST] FEDSWITCH** achieved **90.47%** accuracy in Round 68 (runtime: 192.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDSWITCH** | `practical` | 89.77% | **90.47%** | **91.15%** (R74) | 0.2592 | **90.47%** (R68) | R68 | 0.2845 | 192.4s |