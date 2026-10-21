# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-19 14:34:05  
**Model:** `cnn` | **Seed:** `42` | **Rounds:** 40 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** 🏆 **PROPOSED** achieved **68.01%** accuracy in Round 12 (runtime: 8779.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `cnn` | `adam` | ✅ | **PROPOSED** | `practical` | 27.10% | **68.01%** | **50.24%** (R26) | **68.01%** (R12) | R12 | 58.6851 | 8779.2s |