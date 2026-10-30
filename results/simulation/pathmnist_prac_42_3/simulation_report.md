# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-17 21:09:25  
**Run Name:** `pathmnist_prac_42_3` | **Model:** `baseline` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Disabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** 🏆 **FEDAVG** achieved **80.78%** accuracy in Round 20 (runtime: 5343.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | `sgd` | ✅ | **FEDAVG** | `practical` | 78.55% | **80.78%** | **80.78%** (R20) | **80.78%** (R20) | R20 | 0.9565 | 5343.2s |