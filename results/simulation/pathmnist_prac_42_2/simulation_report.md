# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-17 19:30:30  
**Run Name:** `pathmnist_prac_42_2` | **Model:** `baseline` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Disabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** 🏆 **FEDAVG** achieved **80.63%** accuracy in Round 20 (runtime: 5176.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | `sgd` | ✅ | **FEDAVG** | `practical` | 78.66% | **80.63%** | **80.63%** (R20) | **80.63%** (R20) | R20 | 1.1086 | 5176.8s |