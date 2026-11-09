# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-09 10:00:49  
**Run Name:** `organamnist_practical_fedsol_baseline_bn_1` | **Model:** `baseline_bn` | **Seed:** `42` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = practical:** **[BEST] FEDSOL** achieved **86.34%** accuracy in Round 72 (runtime: 427.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDSOL** | `practical` | 86.15% | **86.34%** | **85.11%** (R72) | 0.5265 | **86.34%** (R72) | R72 | 0.5059 | 427.3s |