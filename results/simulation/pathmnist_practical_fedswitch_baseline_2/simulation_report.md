# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-07 19:46:08  
**Run Name:** `pathmnist_practical_fedswitch_baseline_2` | **Model:** `baseline` | **Seed:** `43` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] FEDSWITCH** achieved **77.14%** accuracy in Round 79 (runtime: 591.7s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | 570,705 | 2229.3 KB | `sgd` | No | **FEDSWITCH** | `practical` | 76.77% | **77.14%** | **75.69%** (R79) | 1.5264 | **77.14%** (R79) | R79 | 1.4137 | 591.7s |