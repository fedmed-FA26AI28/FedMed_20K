# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-19 16:53:16  
**Model:** `baseline` | **Seed:** `42` | **Rounds:** 20 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** 🏆 **PROPOSED** achieved **50.13%** accuracy in Round 8 (runtime: 1252.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | `sgd` | ✅ | **PROPOSED** | `practical` | 18.64% | **50.13%** | **44.45%** (R8) | **50.13%** (R8) | R8 | nan | 1252.3s |