# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-09 09:53:27  
**Run Name:** `organamnist_practical_fedswitch_baseline_bn_5` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **ORGANAMNIST**
- **Partition = practical:** **[BEST] FEDSWITCH** achieved **86.13%** accuracy in Round 79 (runtime: 266.4s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **organamnist** | `baseline_bn` | 571,899 | 2234.0 KB | `sgd` | No | **FEDSWITCH** | `practical` | 86.04% | **86.13%** | **86.42%** (R61) | 0.4688 | **86.13%** (R79) | R79 | 0.5085 | 266.4s |