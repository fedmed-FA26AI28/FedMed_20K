# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-09 10:14:45  
**Run Name:** `organamnist_practical_fedsol_baseline_bn_3` | **Model:** `baseline_bn` | **Seed:** `44` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = practical:** **[BEST] FEDSOL** achieved **86.58%** accuracy in Round 73 (runtime: 383.5s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDSOL** | `practical` | 86.42% | **86.58%** | **86.98%** (R73) | 0.4776 | **86.58%** (R73) | R73 | 0.4952 | 383.5s |