# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-07 17:21:30  
**Run Name:** `audit_fedsol_fix` | **Model:** `baseline` | **Seed:** `42` | **Rounds:** 2 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSOL** achieved **38.82%** accuracy in Round 2 (runtime: 31.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline` | 570,204 | 2227.4 KB | `sgd` | No | **FEDSOL** | `dirichlet_0.3` | 38.82% | **38.82%** | **39.05%** (R2) | 1.8489 | **38.82%** (R2) | R2 | 1.8550 | 31.2s |