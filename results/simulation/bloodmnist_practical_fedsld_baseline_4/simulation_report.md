# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-13 21:50:07  
**Run Name:** `bloodmnist_practical_fedsld_baseline_4` | **Model:** `baseline` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = practical:** **[BEST] FEDSLD** achieved **76.21%** accuracy in Round 78 (runtime: 170.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDSLD** | `practical` | 75.91% | **76.21%** | **75.91%** (R79) | 0.6791 | **76.21%** (R78) | R78 | 0.6791 | 170.3s |