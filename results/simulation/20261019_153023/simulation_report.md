# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-19 15:51:52  
**Model:** `baseline` | **Seed:** `42` | **Rounds:** 20 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** 🏆 **PROPOSED** achieved **47.83%** accuracy in Round 1 (runtime: 1270.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | `adam` | ✅ | **PROPOSED** | `practical` | 25.39% | **47.83%** | **45.69%** (R1) | **47.83%** (R1) | R1 | 2.9757 | 1270.5s |