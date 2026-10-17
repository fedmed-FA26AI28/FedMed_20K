# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-17 18:01:03  
**Model:** `baseline` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Disabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** 🏆 **FEDAVG** achieved **80.15%** accuracy in Round 20 (runtime: 5148.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | `sgd` | ✅ | **FEDAVG** | `practical` | 79.39% | **80.15%** | **80.15%** (R20) | **80.15%** (R20) | R20 | 1.0399 | 5148.6s |