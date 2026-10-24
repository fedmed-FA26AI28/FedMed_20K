# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-24 12:20:00  
**Run Name:** `path_scaf` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `server_decay` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] SCAFFOLD** achieved **82.56%** accuracy in Round 58 (runtime: 1761.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | `sgd` | Yes | **SCAFFOLD** | `practical` | 81.59% | **82.56%** | **80.34%** (R58) | **82.56%** (R58) | R58 | 0.8974 | 1761.3s |