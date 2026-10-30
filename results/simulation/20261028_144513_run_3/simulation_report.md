# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-28 14:47:44  
**Model:** `baseline` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = practical:** **[BEST] FEDPROX** achieved **87.49%** accuracy in Round 77 (runtime: 142.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | `sgd` | No | **FEDPROX** | `practical` | 87.46% | **87.49%** | **86.64%** (R80) | **87.49%** (R77) | R77 | 0.3487 | 142.7s |