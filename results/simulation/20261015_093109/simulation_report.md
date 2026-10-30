# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-15 11:39:56  
**Model:** `baseline` | **Seed:** `41` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = practical:** 🏆 **FEDAVG** achieved **74.96%** accuracy in Round 80 (runtime: 1692.3s).

### Dataset: **PATHMNIST**
- **Partition = practical:** 🏆 **FEDPROX** achieved **75.06%** accuracy in Round 69 (runtime: 4030.8s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline` | `sgd` | ✅ | **FEDAVG** | `practical` | 74.96% | **74.96%** | **73.82%** (R79) | **74.96%** (R80) | R80 | 0.7378 | 1692.3s |
| **organamnist** | `baseline` | `sgd` | ✅ | **FEDPROX** | `practical` | 74.91% | **74.91%** | **73.80%** (R79) | **74.91%** (R80) | R80 | 0.7386 | 1733.8s |
| **pathmnist** | `baseline` | `sgd` | ✅ | **FEDAVG** | `practical` | 0.00% | **0.00%** | N/A | N/A | R0 | 0.0000 | 26.2s |
| **pathmnist** | `baseline` | `sgd` | ✅ | **FEDPROX** | `practical` | 73.06% | **75.06%** | **68.71%** (R69) | **75.06%** (R69) | R69 | 0.9275 | 4030.8s |