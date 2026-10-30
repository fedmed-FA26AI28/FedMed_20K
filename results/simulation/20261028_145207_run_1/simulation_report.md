# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-28 15:04:49  
**Model:** `baseline` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] FEDAVG** achieved **80.25%** accuracy in Round 28 (runtime: 752.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | `sgd` | No | **FEDAVG** | `practical` | 77.99% | **80.25%** | **80.12%** (R28) | **80.25%** (R28) | R28 | 0.9974 | 752.6s |