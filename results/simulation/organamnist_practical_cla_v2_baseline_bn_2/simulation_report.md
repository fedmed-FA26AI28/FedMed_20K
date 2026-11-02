# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-01 23:45:53  
**Run Name:** `organamnist_practical_cla_baseline_bn_2` | **Model:** `baseline_bn` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = practical:** **[BEST] CLA** achieved **85.26%** accuracy in Round 61 (runtime: 342.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **CLA** | `practical` | 85.07% | **85.26%** | **85.27%** (R61) | 0.5247 | **85.26%** (R61) | R61 | 0.5378 | 342.2s |