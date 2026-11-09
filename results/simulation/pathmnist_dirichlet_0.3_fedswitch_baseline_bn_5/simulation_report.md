# FedMedAI Simulation Benchmark Report

**Generated At:** 2026-11-09 17:42:53  
**Run Name:** `pathmnist_dirichlet_0.3_fedswitch_baseline_bn_5` | **Model:** `baseline_bn` | **Seed:** `46` | **Rounds:** 80 | **Clients:** 12 | **Local Epochs:** 5 | **LR Mode:** `fixed` | **Local Test:** `Enabled`  

---

## 1. Executive Summary

### Dataset: **PATHMNIST**
- **Partition = dirichlet_0.3:** **[BEST] FEDSWITCH** achieved **81.39%** accuracy in Round 60 (runtime: 513.3s).

---
## 2. Full Simulation Results Matrix

| Dataset | Model | Params | Weight | Opt | Aug | Strategy | Partition | Final Acc (%) | Best Acc (%) | BMCTA (%) | BMCTA Loss | BTA (%) | Best Round | Final Loss | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pathmnist** | `baseline_bn` | 570,897 | 2230.1 KB | `sgd` | No | **FEDSWITCH** | `dirichlet_0.3` | 75.71% | **81.39%** | **82.46%** (R60) | 0.8765 | **81.39%** (R60) | R60 | 1.2620 | 513.3s |