# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-24 11:41:19  
**Run Name:** `path_bn` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `server_decay` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] FEDBN** achieved **87.17%** accuracy in Round 78 (runtime: 1695.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | `sgd` | Yes | **FEDBN** | `practical` | 85.77% | **87.17%** | **84.25%** (R78) | **87.17%** (R78) | R78 | 0.5129 | 1695.6s |