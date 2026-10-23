# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-23 09:50:32  
**Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] PROPOSED** achieved **80.36%** accuracy in Round 56 (runtime: 5050.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | `sgd` | Yes | **PROPOSED** | `practical` | 76.56% | **80.36%** | **78.64%** (R51) | **80.36%** (R56) | R56 | 1.0571 | 5050.4s |