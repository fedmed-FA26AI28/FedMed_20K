# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-09 09:48:46  
**Run Name:** `organamnist_practical_fedswitch_baseline_bn_4` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = practical:** **[BEST] FEDSWITCH** achieved **86.46%** accuracy in Round 60 (runtime: 296.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDSWITCH** | `practical` | 86.07% | **86.46%** | **87.25%** (R80) | 0.4690 | **86.46%** (R60) | R60 | 0.5074 | 296.6s |