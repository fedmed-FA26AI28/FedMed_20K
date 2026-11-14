# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 10:00:55  
**Run Name:** `bloodmnist_dirichlet_0.1_fedsol_baseline_3` | **Model:** `baseline` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] FEDSOL** achieved **86.61%** accuracy in Round 78 (runtime: 187.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDSOL** | `dirichlet_0.1` | 85.79% | **86.61%** | **87.10%** (R71) | 0.3938 | **86.61%** (R78) | R78 | 0.4041 | 187.7s |