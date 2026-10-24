# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-24 10:21:04  
**Run Name:** `organa_prp` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `server_decay` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = practical:** **[BEST] PROPOSED** achieved **95.04%** accuracy in Round 80 (runtime: 661.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | `sgd` | Yes | **PROPOSED** | `practical` | 95.04% | **95.04%** | **92.46%** (R80) | **95.04%** (R80) | R80 | 0.1723 | 661.0s |