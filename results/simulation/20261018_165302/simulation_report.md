# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-18 18:18:33  
**Model:** `baseline` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Disabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** 🏆 **PROPOSED** achieved **49.96%** accuracy in Round 8 (runtime: 5110.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | `sgd` | ✅ | **PROPOSED** | `practical` | 25.01% | **49.96%** | **49.96%** (R8) | **49.96%** (R8) | R8 | 1.7864 | 5110.0s |