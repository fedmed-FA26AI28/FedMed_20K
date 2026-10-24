# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-24 11:10:49  
**Run Name:** `path_prox` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `server_decay` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] FEDPROX** achieved **81.52%** accuracy in Round 51 (runtime: 1792.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | `sgd` | Yes | **FEDPROX** | `practical` | 80.78% | **81.52%** | **78.68%** (R52) | **81.52%** (R51) | R51 | 0.8081 | 1792.5s |