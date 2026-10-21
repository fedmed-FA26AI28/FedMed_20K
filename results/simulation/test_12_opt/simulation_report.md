# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-19 11:50:13  
**Run Name:** `test_12_opt` | **Model:** `cnn` | **Seed:** `42` | **Rounds:** 1 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** 🏆 **PROPOSED** achieved **17.41%** accuracy in Round 1 (runtime: 244.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `cnn` | `adam` | ✅ | **PROPOSED** | `practical` | 17.41% | **17.41%** | **9.68%** (R1) | **17.41%** (R1) | R1 | 10.4209 | 244.6s |