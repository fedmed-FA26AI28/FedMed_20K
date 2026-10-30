# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-15 04:38:21  
**Model:** `baseline` | **Seed:** `41` | **Rounds:** 160 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = pathological:** 🏆 **FEDAVG** achieved **70.49%** accuracy in Round 160 (runtime: 3433.6s).

### Dataset: **PATHMNIST**
- **Partition = pathological:** 🏆 **FEDPROX** achieved **58.27%** accuracy in Round 148 (runtime: 6209.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | `sgd` | ✅ | **FEDAVG** | `pathological` | 70.49% | **70.49%** | **70.54%** (R160) | **70.49%** (R160) | R160 | 0.8861 | 3433.6s |
| **organamnist** | `baseline` | `sgd` | ✅ | **FEDPROX** | `pathological` | 70.38% | **70.38%** | **70.44%** (R160) | **70.38%** (R160) | R160 | 0.8869 | 3505.7s |
| **pathmnist** | `baseline` | `sgd` | ✅ | **FEDAVG** | `pathological` | 58.11% | **58.18%** | **60.73%** (R148) | **58.18%** (R148) | R148 | 1.3123 | 6081.2s |
| **pathmnist** | `baseline` | `sgd` | ✅ | **FEDPROX** | `pathological` | 58.11% | **58.27%** | **60.82%** (R148) | **58.27%** (R148) | R148 | 1.3128 | 6209.4s |