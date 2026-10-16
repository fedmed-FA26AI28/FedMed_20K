# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-16 10:12:12  
**Model:** `baseline` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** 🏆 **FEDAVG** achieved **80.91%** accuracy in Round 68 (runtime: 4624.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | `sgd` | ✅ | **FEDAVG** | `practical` | 80.75% | **80.91%** | **81.86%** (R68) | **80.91%** (R68) | R68 | 0.8180 | 4624.5s |