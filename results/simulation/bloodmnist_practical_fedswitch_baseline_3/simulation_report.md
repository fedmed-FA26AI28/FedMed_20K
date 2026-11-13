# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-13 22:03:12  
**Run Name:** `bloodmnist_practical_fedswitch_baseline_3` | **Model:** `baseline` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = practical:** **[BEST] FEDSWITCH** achieved **88.02%** accuracy in Round 77 (runtime: 191.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDSWITCH** | `practical` | 87.90% | **88.02%** | **87.74%** (R66) | 0.3497 | **88.02%** (R77) | R77 | 0.3305 | 191.6s |