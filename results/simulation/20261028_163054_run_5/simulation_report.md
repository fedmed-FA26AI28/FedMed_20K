# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-28 16:41:54  
**Model:** `baseline` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] FEDPROX** achieved **79.93%** accuracy in Round 43 (runtime: 651.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | `sgd` | No | **FEDPROX** | `practical` | 78.44% | **79.93%** | **78.97%** (R73) | **79.93%** (R43) | R43 | 0.8560 | 651.0s |