# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-14 12:51:08  
**Run Name:** `bloodmnist_practical_fedbns_baseline_bn_4` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **BLOODMNIST**
- **Partition = practical:** **[BEST] FEDBNS** achieved **97.57%** accuracy in Round 80 (runtime: 169.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bloodmnist** | `baseline_bn` | 570,396 | 2228.1 KB | `sgd` | No | **FEDBNS** | `practical` | 97.57% | **97.57%** | **96.51%** (R77) | 0.1048 | **97.57%** (R80) | R80 | 0.0845 | 169.2s |