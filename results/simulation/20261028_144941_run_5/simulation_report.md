# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-28 14:51:35  
**Model:** `baseline` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = practical:** **[BEST] FEDPROX** achieved **85.53%** accuracy in Round 79 (runtime: 105.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | `sgd` | No | **FEDPROX** | `practical` | 85.30% | **85.53%** | **83.08%** (R79) | **85.53%** (R79) | R79 | 0.4058 | 105.7s |