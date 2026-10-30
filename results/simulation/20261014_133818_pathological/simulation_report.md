# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-14 19:05:13  
**Model:** `baseline` | **Seed:** `41` | **Rounds:** 160 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = pathological:** 🏆 **FEDAVG** achieved **70.47%** accuracy in Round 160 (runtime: 3544.9s).

### Dataset: **PATHMNIST**
- **Partition = pathological:** 🏆 **FEDAVG** achieved **58.16%** accuracy in Round 148 (runtime: 6054.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | `sgd` | ✅ | **FEDAVG** | `pathological` | 70.47% | **70.47%** | **70.52%** (R160) | **70.47%** (R160) | R160 | 0.8853 | 3544.9s |
| **organamnist** | `baseline` | `sgd` | ✅ | **FEDPROX** | `pathological` | 69.97% | **69.97%** | **69.96%** (R160) | **69.97%** (R160) | R160 | 0.8979 | 3566.7s |
| **pathmnist** | `baseline` | `sgd` | ✅ | **FEDAVG** | `pathological` | 57.99% | **58.16%** | **60.65%** (R148) | **58.16%** (R148) | R148 | 1.3171 | 6054.4s |
| **pathmnist** | `baseline` | `sgd` | ✅ | **FEDPROX** | `pathological` | 57.16% | **57.26%** | **59.83%** (R149) | **57.26%** (R149) | R149 | 1.2807 | 6191.1s |