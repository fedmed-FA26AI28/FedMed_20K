# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-24 09:57:42  
**Run Name:** `organa_bn_scaf` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `server_decay` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = practical:** **[BEST] FEDBN** achieved **94.53%** accuracy in Round 80 (runtime: 644.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | `sgd` | Yes | **FEDBN** | `practical` | 94.53% | **94.53%** | **91.75%** (R79) | **94.53%** (R80) | R80 | 0.1789 | 644.4s |
| **organamnist** | `baseline_bn` | `sgd` | Yes | **SCAFFOLD** | `practical` | 88.53% | **88.57%** | **87.60%** (R73) | **88.57%** (R58) | R58 | 0.3416 | 745.5s |