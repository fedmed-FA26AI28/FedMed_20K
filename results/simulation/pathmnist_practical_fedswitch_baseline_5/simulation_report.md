# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-07 20:19:35  
**Run Name:** `pathmnist_practical_fedswitch_baseline_5` | **Model:** `baseline` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] FEDSWITCH** achieved **78.27%** accuracy in Round 43 (runtime: 665.6s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline` | 570,705 | 2229.3 KB | `sgd` | No | **FEDSWITCH** | `practical` | 76.94% | **78.27%** | **77.34%** (R69) | 0.9207 | **78.27%** (R43) | R43 | 0.9278 | 665.6s |