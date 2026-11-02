# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-02 00:02:44  
**Run Name:** `organamnist_practical_cla_baseline_bn_5` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = practical:** **[BEST] CLA** achieved **83.91%** accuracy in Round 77 (runtime: 303.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **CLA** | `practical` | 83.87% | **83.91%** | **84.50%** (R80) | 0.5757 | **83.91%** (R77) | R77 | 0.6008 | 303.3s |