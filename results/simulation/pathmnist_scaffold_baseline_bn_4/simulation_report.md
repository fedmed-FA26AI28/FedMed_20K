# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-10-30 12:07:54  
**Run Name:** `pathmnist_scaffold_baseline_bn_4` | **Model:** `baseline_bn` | **Seed:** `45` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = practical:** **[BEST] SCAFFOLD** achieved **84.50%** accuracy in Round 54 (runtime: 768.2s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **SCAFFOLD** | `practical` | 81.25% | **84.50%** | **83.76%** (R54) | 0.8627 | **84.50%** (R54) | R54 | 0.9238 | 768.2s |