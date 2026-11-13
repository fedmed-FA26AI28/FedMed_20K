# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-13 21:36:22  
**Run Name:** `bloodmnist_practical_fedprox_baseline_5` | **Model:** `baseline` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = practical:** **[BEST] FEDPROX** achieved **85.68%** accuracy in Round 78 (runtime: 172.0s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDPROX** | `practical` | 83.16% | **85.68%** | **83.16%** (R80) | 0.4423 | **85.68%** (R78) | R78 | 0.4423 | 172.0s |