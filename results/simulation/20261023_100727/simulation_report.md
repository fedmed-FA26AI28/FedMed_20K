# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-23 11:33:20  
**Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] FEDAVG** achieved **82.49%** accuracy in Round 65 (runtime: 5130.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | `sgd` | Yes | **FEDAVG** | `practical` | 80.81% | **82.49%** | **80.96%** (R69) | **82.49%** (R65) | R65 | 1.1893 | 5130.5s |