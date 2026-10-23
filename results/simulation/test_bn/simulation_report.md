# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-23 19:44:51  
**Run Name:** `test_bn` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `server_decay` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = practical:** **[BEST] FEDBN** achieved **94.54%** accuracy in Round 80 (runtime: 644.5s).

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] FEDBN** achieved **0.00%** accuracy in Round 0 (runtime: 180.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | `sgd` | Yes | **FEDBN** | `practical` | 94.54% | **94.54%** | **91.80%** (R77) | **94.54%** (R80) | R80 | 0.1776 | 644.5s |
| **pathmnist** | `baseline_bn` | `sgd` | Yes | **FEDBN** | `practical` | 0.00% | **0.00%** | N/A | N/A | R0 | 0.0000 | 180.0s |