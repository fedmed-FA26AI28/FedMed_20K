# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-28 14:38:33  
**Model:** `baseline` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = practical:** **[BEST] FEDAVG** achieved **85.15%** accuracy in Round 78 (runtime: 100.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | `sgd` | No | **FEDAVG** | `practical` | 85.12% | **85.15%** | **84.84%** (R78) | **85.15%** (R78) | R78 | 0.3980 | 100.4s |