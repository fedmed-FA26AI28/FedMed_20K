# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 14:46:47  
**Run Name:** `bloodmnist_dirichlet_0.1_scaffold_baseline_bn_4` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = dirichlet_0.1:** **[BEST] SCAFFOLD** achieved **88.10%** accuracy in Round 80 (runtime: 172.9s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **SCAFFOLD** | `dirichlet_0.1` | 88.10% | **88.10%** | **83.89%** (R80) | 0.4295 | **88.10%** (R80) | R80 | 0.3250 | 172.9s |